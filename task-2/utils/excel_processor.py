"""
Excel to JSON Processor Module
A dynamic, modular system for converting Excel files to JSON formats.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
import re
from pathlib import Path
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Enum for different data types in Excel sheets"""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATE = "date"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class ColumnSchema:
    """
    Schema definition for a column in Excel sheet

    Attributes:
        name (str): Column name
        data_type (DataType): Expected data type
        required (bool): Whether column is required
        description (str): Column description
        validations (List[str]): List of validation rules
    """

    name: str
    data_type: DataType = DataType.UNKNOWN
    required: bool = False
    description: str = ""
    validations: List[str] = field(default_factory=list)


@dataclass
class SheetConfig:
    """
    Configuration for processing an Excel sheet

    Attributes:
        sheet_name (str): Name of the sheet in Excel file
        data_start_row (int): Row index where data begins (0-indexed)
        header_rows (List[int]): List of row indices containing headers (0-indexed)
        skip_rows (List[int]): Rows to skip (e.g., subtotal rows)
        data_end_marker (str): Marker indicating end of data (e.g., "TOTAL", "جمع")
        column_schemas (List[ColumnSchema]): Expected column schemas (optional)
        sheet_type (str): Type of sheet for special processing
        date_format (str): Expected date format if dates are present
    """

    sheet_name: str
    data_start_row: int
    header_rows: List[int]
    skip_rows: List[int] = field(default_factory=list)
    data_end_marker: Optional[str] = None
    column_schemas: List[ColumnSchema] = field(default_factory=list)
    sheet_type: str = "general"
    date_format: str = "%Y-%m-%d"


class ExcelToJSONProcessor:
    """
    Main processor class for converting Excel files to JSON format

    This class provides:
    - Dynamic sheet detection and processing
    - Intelligent data type inference
    - Flexible configuration system
    - Comprehensive error handling
    - Support for Persian/Arabic characters
    - Batch processing capabilities
    """

    # Default configurations
    DEFAULT_CONFIGS = {
        # Stock investments sheet
        "ایران زمین ثابت": SheetConfig(
            sheet_name="سرمایه گذاری در سهام",
            data_start_row=10,  # Excel row 11
            header_rows=[8, 9],  # Excel rows 9, 10
            skip_rows=[39],  # Skip total row
            data_end_marker="جمع",
            sheet_type="stock_investments",
        ),
        "اطلس": SheetConfig(
            sheet_name="سهام",
            data_start_row=8,
            header_rows=[6, 7],
            skip_rows=[95],
            data_end_marker="",
            sheet_type="stock_investments",
        ),
        "عقیق سهام": SheetConfig(
            sheet_name="سهام و حق تقدم سهام",
            data_start_row=10,
            header_rows=[8, 9],
            skip_rows=[92],
            data_end_marker="جمع",
            sheet_type="stock_investments",
        ),
    }

    def __init__(self, file_path: Union[str, Path], auto_load: bool = True):
        """
        Initialize Excel processor

        Args:
            file_path (Union[str, Path]): Path to Excel file
            auto_load (bool): Whether to automatically load the file on init
        """
        self.file_path = Path(file_path)
        self.workbook = None
        self.configs = self.DEFAULT_CONFIGS.copy()
        self._validated_sheets = {}

        if auto_load:
            self.load_workbook()

    def load_workbook(self) -> bool:
        """
        Load Excel workbook into memory

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.file_path.exists():
                logger.error(f"File not found: {self.file_path}")
                return False

            self.workbook = pd.ExcelFile(self.file_path)
            logger.info(f"Successfully loaded workbook: {self.file_path.name}")
            logger.info(f"Available sheets: {', '.join(self.workbook.sheet_names)}")
            return True

        except Exception as e:
            logger.error(f"Failed to load workbook: {e}")
            return False

    def add_sheet_config(self, config: SheetConfig) -> None:
        """
        Add or update configuration for a sheet

        Args:
            config (SheetConfig): Configuration object for the sheet
        """
        self.configs[config.sheet_name] = config
        logger.info(f"Added/updated config for sheet: {config.sheet_name}")

    def get_sheet_config(self, symbol: str) -> Optional[SheetConfig]:
        """
        Get configuration for a specific symbol

        Args:
            symbol (str): Name of the symbol

        Returns:
            Optional[SheetConfig]: Configuration if found, None otherwise
        """
        return self.configs.get(symbol)

    def detect_sheet_type(self, df: pd.DataFrame) -> str:
        """
        Automatically detect sheet type based on content

        Args:
            df (pd.DataFrame): DataFrame of sheet content

        Returns:
            str: Detected sheet type
        """
        # Get first few non-null values from first column
        sample_data = df.iloc[:10, 0].dropna().astype(str).tolist()

        # Define patterns for different sheet types
        type_patterns = {
            "stock": ["سهام", "شرکت", "آريان", "ايران خودرو"],
            "fund": ["صندوق", "سرمایه گذاری"],
            "bond": ["اوراق", "اسناد", "مرابحه", "صكوك"],
            "bank": ["سپرده", "بانک", "بانكي"],
            "derivative": ["اختيار", "تبعي", "مشتقه"],
        }

        for sheet_type, patterns in type_patterns.items():
            for pattern in patterns:
                if any(pattern in cell for cell in sample_data):
                    return sheet_type

        return "unknown"

    def clean_column_name(self, name: str) -> str:
        """
        Clean and normalize column names

        Args:
            name (str): Original column name

        Returns:
            str: Cleaned column name in snake_case
        """
        if pd.isna(name) or not str(name).strip():
            return ""

        # Convert to string and clean
        clean_name = str(name)

        # Remove special characters
        clean_name = re.sub(r'[‫″"\'\t\n\r]', "", clean_name)

        # Replace problematic characters with underscore
        clean_name = re.sub(r"[/\\:;?!@#$%^&*()+=\[\]{}|<>~`]", "_", clean_name)

        # Replace Persian/Arabic spaces and dashes
        clean_name = re.sub(r"[\s\u200c\u200f\-]+", "_", clean_name)

        # Convert to lowercase and remove duplicate underscores
        clean_name = clean_name.lower()
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")

        return clean_name

    def infer_data_type(self, value: Any) -> DataType:
        """
        Infer data type from a value

        Args:
            value (Any): Input value

        Returns:
            DataType: Inferred data type
        """
        if pd.isna(value):
            return DataType.UNKNOWN

        # Check for numeric types
        if isinstance(value, (int, np.integer)):
            return DataType.INTEGER
        elif isinstance(value, (float, np.floating)):
            return DataType.FLOAT

        # Convert to string for further analysis
        str_value = str(value).strip()

        # Check for empty string
        if not str_value:
            return DataType.UNKNOWN

        # Check for numbers stored as strings
        num_pattern = r"^-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?$"
        if re.match(num_pattern, str_value.replace(",", "")):
            if "." in str_value or "e" in str_value.lower() or "E" in str_value:
                return DataType.FLOAT
            else:
                return DataType.INTEGER

        # Check for boolean-like values
        bool_values = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
            "صحیح": True,
            "غلط": False,
        }
        if str_value.lower() in bool_values:
            return DataType.BOOLEAN

        # Check for date patterns (basic detection)
        date_patterns = [
            r"\d{4}[-/]\d{2}[-/]\d{2}",  # YYYY-MM-DD
            r"\d{2}[-/]\d{2}[-/]\d{4}",  # DD-MM-YYYY
        ]
        for pattern in date_patterns:
            if re.match(pattern, str_value):
                return DataType.DATE

        return DataType.STRING

    def normalize_value(self, value: Any, target_type: DataType = None) -> Any:
        """
        Normalize a value to appropriate Python type

        Args:
            value (Any): Input value
            target_type (DataType, optional): Target data type

        Returns:
            Any: Normalized value
        """
        if pd.isna(value):
            return None

        # If no target type specified, infer it
        if target_type is None:
            target_type = self.infer_data_type(value)

        str_value = str(value).strip()

        try:
            if target_type == DataType.INTEGER:
                # Remove thousand separators and convert
                clean = str_value.replace(",", "").replace("،", "")
                return int(float(clean)) if "." in clean else int(clean)

            elif target_type == DataType.FLOAT:
                # Handle Persian/Arabic decimal separators
                clean = str_value.replace(",", "").replace("،", "")
                return float(clean)

            elif target_type == DataType.BOOLEAN:
                bool_map = {
                    "true": True,
                    "false": False,
                    "yes": True,
                    "no": False,
                    "1": True,
                    "0": False,
                    "صحیح": True,
                    "غلط": False,
                    "درست": True,
                    "نادرست": False,
                }
                return bool_map.get(str_value.lower(), bool(str_value))

            elif target_type == DataType.DATE:
                # Attempt to parse date (basic implementation)
                # For production, use proper date parsing library
                return str_value

            else:  # STRING or UNKNOWN
                return str_value if str_value else None

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to normalize value '{value}' to {target_type}: {e}")
            return str_value if str_value else None

    def extract_headers(self, df: pd.DataFrame, config: SheetConfig) -> List[str]:
        """
        Extract column headers from specified rows

        Args:
            df (pd.DataFrame): DataFrame of sheet content
            config (SheetConfig): Sheet configuration

        Returns:
            List[str]: List of cleaned header names
        """
        headers = []

        for col_idx in range(df.shape[1]):
            header_parts = []

            # Combine header parts from multiple rows
            for row_idx in config.header_rows:
                if row_idx < len(df):
                    cell_value = df.iloc[row_idx, col_idx]
                    if pd.notna(cell_value):
                        header_parts.append(str(cell_value))

            # Join header parts with separator
            if header_parts:
                full_header = " - ".join(list(set(header_parts)))
            else:
                full_header = f"column_{col_idx}"

            # Clean the header name
            clean_header = self.clean_column_name(full_header)
            headers.append(clean_header)

        logger.debug(f"Extracted {len(headers)} headers")
        return headers

    def extract_data_rows(
        self, df: pd.DataFrame, config: SheetConfig
    ) -> List[Dict[str, Any]]:
        """
        Extract data rows from DataFrame based on configuration

        Args:
            df (pd.DataFrame): DataFrame of sheet content
            config (SheetConfig): Sheet configuration

        Returns:
            List[Dict[str, Any]]: List of data rows as dictionaries
        """
        data_rows = []
        headers = self.extract_headers(df, config)

        # Process each data row
        for row_idx in range(config.data_start_row, len(df)):
            # Check for end marker
            if (
                config.data_end_marker
                and pd.notna(df.iloc[row_idx, 0])
                and config.data_end_marker in str(df.iloc[row_idx, 0])
            ):
                logger.debug(f"Found end marker at row {row_idx}")
                break

            # Skip specified rows
            if row_idx in config.skip_rows:
                continue

            # Extract row data
            row_data = {}
            has_valid_data = False

            for col_idx, header in enumerate(headers):
                if col_idx < df.shape[1]:
                    value = df.iloc[row_idx, col_idx]

                    # Only process if we have a header and value is not entirely empty
                    if header and (pd.notna(value) or config.column_schemas):
                        # Get target type from schema if available
                        target_type = None
                        if config.column_schemas and col_idx < len(
                            config.column_schemas
                        ):
                            target_type = config.column_schemas[col_idx].data_type

                        normalized_value = self.normalize_value(value, target_type)

                        if normalized_value is not None:
                            row_data[header] = normalized_value
                            has_valid_data = True

            # Only add row if it has valid data
            if has_valid_data:
                data_rows.append(row_data)

        logger.info(f"Extracted {len(data_rows)} data rows from sheet")
        return data_rows

    def validate_data_row(
        self, row: Dict[str, Any], schemas: List[ColumnSchema]
    ) -> Dict[str, List[str]]:
        """
        Validate a data row against column schemas

        Args:
            row (Dict[str, Any]): Data row to validate
            schemas (List[ColumnSchema]): Column schemas

        Returns:
            Dict[str, List[str]]: Validation results (errors by column)
        """
        errors = {}

        for i, schema in enumerate(schemas):
            col_name = schema.name
            value = row.get(col_name)

            # Check required columns
            if schema.required and (value is None or value == ""):
                errors.setdefault(col_name, []).append("Required field is missing")

            # Type validation
            if value is not None and value != "":
                expected_type = schema.data_type
                actual_type = self.infer_data_type(value)

                if expected_type != DataType.UNKNOWN and actual_type != expected_type:
                    errors.setdefault(col_name, []).append(
                        f"Type mismatch: expected {expected_type.value}, got {actual_type.value}"
                    )

            # Custom validations
            for validation in schema.validations:
                # This is a simple implementation - extend as needed
                if (
                    validation == "positive"
                    and isinstance(value, (int, float))
                    and value < 0
                ):
                    errors.setdefault(col_name, []).append("Value must be positive")

        return errors

    def process_sheet(
        self, symbol: str, config: Optional[SheetConfig] = None
    ) -> Dict[str, Any]:
        """
        Process a single sheet and convert to structured JSON

        Args:
            symbol (str): Name of symbol to process
            config (Optional[SheetConfig]): Sheet configuration (uses default if None)

        Returns:
            Dict[str, Any]: Processed data with metadata
        """
        logger.info(f"Processing sheet of symbol: {symbol}")

        # Get configuration
        if not config:
            config = self.get_sheet_config(symbol)
            if not config:
                logger.warning(f"No configuration found for this symbol: {symbol}")
                return self._create_empty_result(symbol, "No configuration")

        try:
            sheet_name = self.configs.get(symbol).sheet_name
            # Read sheet without headers (we'll extract them manually)
            df = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                header=None,
                dtype=str,  # Read everything as string initially
                engine="openpyxl",
            )

            # Extract headers and data
            headers = self.extract_headers(df, config)
            data_rows = self.extract_data_rows(df, config)

            # Validate data if schemas are provided
            validation_results = {}
            if config.column_schemas:
                for i, row in enumerate(data_rows):
                    errors = self.validate_data_row(row, config.column_schemas)
                    if errors:
                        validation_results[f"row_{i}"] = errors

            # Create result structure
            result = {
                "metadata": {
                    "sheet_name": sheet_name,
                    "sheet_type": config.sheet_type,
                    "total_records": len(data_rows),
                    "total_columns": len(headers),
                    "processing_timestamp": pd.Timestamp.now().isoformat(),
                    "config_used": {
                        "data_start_row": config.data_start_row,
                        "header_rows": config.header_rows,
                        "has_schemas": bool(config.column_schemas),
                    },
                },
                # "columns": headers,
                "data": data_rows,
                "validation": (
                    {
                        "has_errors": bool(validation_results),
                        "error_count": sum(
                            len(errors) for errors in validation_results.values()
                        ),
                        "errors": validation_results,
                    }
                    if config.column_schemas
                    else None
                ),
            }

            logger.info(
                f"Successfully processed sheet ({sheet_name}) of symbol '{symbol}': {len(data_rows)} records"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing sheet of symbol '{symbol}': {e}")
            return self._create_empty_result(symbol, str(e))

    def _create_empty_result(
        self, symbol: str, error_message: str
    ) -> Dict[str, Any]:
        """Create an empty result structure for failed processing"""
        return {
            "metadata": {
                "sheet_name": self.configs.get(symbol).sheet_name,
                "error": error_message,
                "total_records": 0,
                "processing_timestamp": pd.Timestamp.now().isoformat(),
            },
            "data": [],
        }

    # def process_all_sheets(self) -> Dict[str, Dict[str, Any]]:
    #     """
    #     Process all sheets in the workbook

    #     Returns:
    #         Dict[str, Dict[str, Any]]: Dictionary of processed sheets (sheet_name -> result)
    #     """
    #     if not self.workbook:
    #         logger.error("Workbook not loaded. Call load_workbook() first.")
    #         return {}

    #     all_results = {}
    #     logger.info(f"Processing all sheets in workbook")

    #     for sheet_name in self.workbook.sheet_names:
    #         result = self.process_sheet(sheet_name)
    #         if result:
    #             all_results[sheet_name] = result

    #     # Generate summary statistics
    #     total_records = sum(
    #         r["metadata"]["total_records"] for r in all_results.values()
    #     )
    #     logger.info(
    #         f"Processing complete. Total records across all sheets: {total_records}"
    #     )

    #     return all_results

    def process_selected_sheets(
        self, sheet_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process only selected sheets

        Args:
            sheet_names (List[str]): List of sheet names to process

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of processed sheets
        """
        results = {}

        for sheet_name in sheet_names:
            if sheet_name in self.workbook.sheet_names:
                result = self.process_sheet(sheet_name)
                if result:
                    results[sheet_name] = result
            else:
                logger.warning(f"Sheet '{sheet_name}' not found in workbook")

        return results

    def save_to_json(
        self, data: Dict, output_file: Union[str, Path], indent: int = 2
    ) -> bool:
        """
        Save processed data to JSON file

        Args:
            data (Dict): Data to save
            output_file (Union[str, Path]): Output file path
            indent (int): JSON indentation level

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent, default=str)

            logger.info(f"Data saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}")
            return False

    def export_individual_sheets(
        self, data: Dict[str, Dict], output_dir: Union[str, Path]
    ) -> None:
        """
        Export each sheet to separate JSON file

        Args:
            data (Dict[str, Dict]): Dictionary of sheet data
            output_dir (Union[str, Path]): Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting individual sheets to: {output_dir}")

        for sheet_name, sheet_data in data.items():
            # Create safe filename
            safe_name = re.sub(r"[^\w\-_\. ]", "_", sheet_name)
            filename = output_dir / f"{safe_name}.json"

            self.save_to_json(sheet_data, filename)

        logger.info(f"Exported {len(data)} sheets to individual files")

    def analyze_sheet_structure(self, sheet_name: str) -> Dict[str, Any]:
        """
        Analyze sheet structure to help with configuration

        Args:
            sheet_name (str): Name of sheet to analyze

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            df = pd.read_excel(
                self.file_path, sheet_name=sheet_name, header=None, nrows=50
            )

            analysis = {
                "sheet_name": sheet_name,
                "total_rows": len(df),
                "total_columns": df.shape[1],
                "sample_data": {},
                "recommended_config": {},
            }

            # Analyze first 10 rows
            for i in range(min(15, len(df))):
                row_data = {}
                for j in range(min(10, df.shape[1])):
                    cell = df.iloc[i, j]
                    if pd.notna(cell):
                        row_data[f"col_{j}"] = {
                            "value": str(cell)[:50],  # First 50 chars
                            "type": type(cell).__name__,
                        }
                if row_data:
                    analysis["sample_data"][f"row_{i}"] = row_data

            # Try to find headers
            header_candidates = []
            for i in range(min(20, len(df))):
                row = df.iloc[i]
                # Count non-empty cells in row
                non_empty = sum(
                    1 for cell in row if pd.notna(cell) and str(cell).strip()
                )
                if non_empty > 0:
                    header_candidates.append(
                        {
                            "row": i,
                            "non_empty_cells": non_empty,
                            "sample": [
                                str(cell)[:20] for cell in row[:5] if pd.notna(cell)
                            ],
                        }
                    )

            analysis["header_candidates"] = header_candidates

            # Generate recommended config
            if header_candidates:
                best_candidate = max(
                    header_candidates, key=lambda x: x["non_empty_cells"]
                )
                analysis["recommended_config"] = {
                    "data_start_row": best_candidate["row"] + 1,
                    "header_rows": [best_candidate["row"]],
                    "note": "Auto-generated recommendation. Adjust as needed.",
                }

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze sheet '{sheet_name}': {e}")
            return {"error": str(e)}
