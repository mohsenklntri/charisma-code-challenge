"""
Main runner for Excel to JSON Processor
"""

import sys
import argparse
from pathlib import Path
from .excel_processor import ExcelToJSONProcessor


def main():
    """Main entry point with command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Excel to JSON Converter - Process Excel files to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """,
    )

    parser.add_argument("file", help="Path to Excel file")
    parser.add_argument(
        "-s", "--symbol", 
        choices=["اطلس", "ایران زمین ثابت", "عقیق سهام"],
    )
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set logging level
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Check if file exists
    if not Path(args.file).exists():
        print(f"❌ Error: File '{args.file}' not found")
        sys.exit(1)

    # Create processor
    processor = ExcelToJSONProcessor(args.file)
    
    process_specific_sheet(processor, args.symbol, args.output)


def list_sheets(processor):
    """List all sheets in workbook"""
    if not processor.workbook:
        print("❌ Workbook not loaded")
        return

    print("\n📋 SHEETS IN WORKBOOK:")
    for i, sheet in enumerate(processor.workbook.sheet_names, 1):
        config = processor.get_sheet_config(sheet)
        config_status = "✅" if config else "⚠️ "
        print(f"  {i}. {config_status} {sheet}")


def process_specific_sheet(processor, symbol, output_dir):
    """Process specific sheets from command line"""
    result = processor.process_sheet(symbol)

    if result:
        sheet_name = result["metadata"]["sheet_name"]
        output_file = Path(output_dir) / f"{symbol}.json"
        processor.save_to_json(result, output_file)
        print(f"✅ Data saved to: {output_file}")
        
        records = result["metadata"]["total_records"]
        print(f"📈 {sheet_name}: {records} records")
    else:
        print("❌ No data processed")


if __name__ == "__main__":
    main()
