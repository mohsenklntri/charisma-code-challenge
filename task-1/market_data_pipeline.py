"""
Market Data Pipeline Simulator
Simulates NiFi flow: API → Split JSON → Kafka
"""

import time
import json
import requests
from kafka import KafkaProducer
from typing import List, Dict
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataPipeline:
    def __init__(self, api_url: str, kafka_bootstrap: str, topic: str = "market_map"):
        """
        Initialize the pipeline
        
        Args:
            api_url: API endpoint to fetch data from
            kafka_bootstrap: Kafka bootstrap servers (e.g., "localhost:9092")
            topic: Kafka topic name
        """
        self.api_url = api_url
        self.topic = topic
        
        # Initialize Kafka producer
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda v: str(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info(f"Connected to Kafka at {kafka_bootstrap}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            sys.exit(1)
    
    def fetch_market_data(self) -> List[Dict]:
        """
        Simulate InvokeHTTP processor - Fetch data from API
        
        Returns:
            List of market data records
        """
        try:
            # Simulate API request
            response = requests.get(
                self.api_url,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'MarketDataPipeline/1.0'
                },
                timeout=5
            )
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Assume API returns an array
            if isinstance(data, list):
                logger.info(f"Fetched {len(data)} records from API")
                return data
            elif isinstance(data, dict) and 'data' in data:
                # Handle wrapped response
                logger.info(f"Fetched {len(data['data'])} records from API")
                return data['data']
            else:
                logger.warning("API response format unexpected")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []
    
    def process_record(self, record: Dict) -> Dict:
        """
        Simulate simple transformation in Produce Data process group
        
        Args:
            record: Raw market data record
            
        Returns:
            Processed record
        """
        processed = record.copy()

        processed['_processed_at'] = datetime.now().isoformat()
        
        if 'customLabel' not in processed:
            logger.warning(f"Record missing 'customLabel' field: {record}")
        
        return processed
    
    def publish_to_kafka(self, records: List[Dict]):
        """
        Simulate PublishKafka processor
        
        Args:
            records: List of processed records to publish
        """
        success_count = 0
        
        for record in records:
            try:
                # Use symbol as Kafka key for partitioning
                key = record.get('customLabel', 'unknown')
                
                # Send to Kafka
                future = self.producer.send(
                    topic=self.topic,
                    key=key,
                    value=record
                )
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to publish record {record.get('symbol')}: {e}")
        
        logger.info(f"Published {success_count}/{len(records)} records to Kafka topic '{self.topic}'")
    
    def run_single_cycle(self):
        """
        Run one complete cycle of the pipeline
        """
        logger.info("=" * 50)
        logger.info("Starting pipeline cycle")
        
        # Step 1: Get Data (API request)
        logger.info("Step 1: Fetching data from API...")
        records = self.fetch_market_data()
        
        if not records:
            logger.warning("No data received from API")
            return
        
        # Step 2: Split JSON (implicit - we already have list)
        logger.info(f"Step 2: Processing {len(records)} individual records")
        
        # Step 3: Process each record
        processed_records = []
        for i, record in enumerate(records):
            processed = self.process_record(record)
            processed_records.append(processed)
            logger.debug(f"Processed record {i+1}: {record.get('symbol', 'unknown')}")
        
        # Step 4: Publish to Kafka
        logger.info(f"Step 4: Publishing to Kafka topic '{self.topic}'...")
        self.publish_to_kafka(processed_records)
        
        logger.info("Pipeline cycle completed")
    
    def run_continuously(self, interval_seconds: int = 1):
        """
        Run the pipeline continuously with specified interval
        
        Args:
            interval_seconds: Time between cycles (default: 1 second)
        """
        logger.info(f"Starting continuous pipeline (interval: {interval_seconds}s)")
        logger.info(f"API: {self.api_url}")
        logger.info(f"Kafka Topic: {self.topic}")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"Cycle #{cycle_count}")
                
                start_time = time.time()
                self.run_single_cycle()
                end_time = time.time()
                
                # Calculate sleep time to maintain 1-second interval
                elapsed = end_time - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Cycle took {elapsed:.2f}s (longer than {interval_seconds}s interval)")
                    
        except KeyboardInterrupt:
            logger.info("\nPipeline stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        if hasattr(self, 'producer'):
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")
    
def main():
    """Main function with configuration options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Data Pipeline Simulator')
    parser.add_argument('--api-url', default='https://cdn.tsetmc.com/api/ClosingPrice/GetMarketMap?market=TseStock-OtcStock-&size=789&sector=0&typeSelected=1&hEven=180557',
                       help='API endpoint URL (default: mock API)')
    parser.add_argument('--kafka', default='localhost:9092',
                       help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--topic', default='market_map',
                       help='Kafka topic name (default: market_map)')
    parser.add_argument('--interval', type=int, default=1,
                       help='Interval between cycles in seconds (default: 1)')
    parser.add_argument('--single', action='store_true',
                       help='Run only one cycle and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = MarketDataPipeline(
        api_url=args.api_url,
        kafka_bootstrap=args.kafka,
        topic=args.topic
    )
    
    if args.single:
        pipeline.run_single_cycle()
        pipeline.cleanup()
    else:
        pipeline.run_continuously(interval_seconds=args.interval)


if __name__ == "__main__":
    main()