"""CSV processing with LLM-powered data generation."""

import json
import logging
from pathlib import Path
from time import time
from typing import List, Optional, Type

import pandas as pd
from pydantic import BaseModel

from .config import DEFAULT_PREVIEW_ROWS
from .llm_client import LLMClient
from .schemas import get_field_names
from .utils import color_text, format_error_message

logger = logging.getLogger(__name__)


class CSVProcessor:
    """Process CSV files with LLM to generate new structured columns."""

    def __init__(self, llm_client: LLMClient):
        """Initialize the CSV processor.

        Args:
            llm_client: An initialized LLM client instance.
        """
        self.llm_client = llm_client

    def _should_process_row(self, row: pd.Series, new_column_names: List[str]) -> bool:
        """Check if a row should be processed (has empty values in new columns).

        Args:
            row: DataFrame row
            new_column_names: List of new column names to check

        Returns:
            True if row should be processed, False if already filled
        """
        for col in new_column_names:
            if col in row.index:
                value = row[col]
                # Check if value is missing or empty
                if pd.isna(value) or value == "":
                    return True
                # Try to parse as JSON and check if empty
                try:
                    parsed = json.loads(str(value))
                    if not parsed or parsed == {}:
                        return True
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Not valid JSON, keep existing value
                    pass
        # If none of the columns exist, we should process
        if not any(col in row.index for col in new_column_names):
            return True
        # All columns have data
        return False

    def process_csv(
        self,
        input_path: Path | str,
        output_path: Path | str,
        columns: List[str],
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        resume: bool = True,
    ) -> None:
        """Process a CSV file and add LLM-generated columns.

        Args:
            input_path: Path to the input CSV file.
            output_path: Path to the output CSV file.
            columns: List of column names to use as input for LLM.
            prompt: User-provided prompt describing the task.
            response_model: Optional Pydantic model for structured output.
            resume: If True and output exists, skip rows with data.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If specified columns don't exist in CSV.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if response_model is None:
            raise ValueError(
                "Output schema is required. Dynamic schema is not supported."
            )

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load CSV - use output if it exists and resume is True
        if output_path.exists() and resume:
            logger.info("Loading existing output (resume): %s", output_path)
            df = pd.read_csv(output_path)
            logger.info("Loaded %d rows (resume mode)", len(df))
        else:
            logger.info("Loading CSV from %s", input_path)
            df = pd.read_csv(input_path)
            logger.info("Loaded %d rows", len(df))

        # Validate columns
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Columns not found in CSV: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Determine new column names
        new_column_names = get_field_names(response_model)

        # Process each row
        schema_info = ",".join(new_column_names)
        logger.info(
            "Processing rows | model=%s | input=%s | output=%s",
            self.llm_client.model_name,
            ",".join(columns),
            schema_info,
        )
        logger.info("Prompt: %s", prompt)

        results = []
        failed_rows = []
        processed = 0
        skipped = 0
        total_rows = len(df)
        start_time = time()

        for idx, row in df.iterrows():
            row_num = idx + 1
            result_position = len(results)
            prefix = f"[{row_num}/{total_rows}]"

            # Check if row should be skipped (resume mode)
            if (
                resume
                and new_column_names
                and not self._should_process_row(row, new_column_names)
            ):
                logger.info(color_text(f"{prefix} → skip (already processed)", "cyan"))
                # Keep existing data
                existing_data = {
                    col: row[col] if col in row.index else None
                    for col in new_column_names
                }
                results.append(existing_data if existing_data else {})
                skipped += 1
                continue

            try:
                # Extract input data from selected columns
                input_data = {col: row[col] for col in columns}

                # Generate structured output
                output = self.llm_client.generate_structured_output(
                    prompt, input_data, response_model
                )

                results.append(output)
                processed += 1
                logger.info(color_text(f"{prefix} ✓ success", "green"))

            except Exception as e:
                logger.error(color_text(f"{prefix} ✗ {format_error_message(e)}", "red"))
                # Add empty result to maintain row alignment
                results.append({})
                processed += 1
                failed_rows.append(
                    {
                        "position": result_position,
                        "row": row.copy(),
                        "row_num": row_num,
                        "error": format_error_message(e),
                    }
                )

        # Attempt fallback retries for failed rows before finalizing results
        if failed_rows:
            failed_rows = self._retry_failed_rows(
                failed_rows, results, columns, prompt, response_model
            )

        errors = [f"Row {info['row_num']}: {info['error']}" for info in failed_rows]

        # Convert results to DataFrame columns
        if results:
            # Flatten nested JSON into columns
            result_df = pd.json_normalize(results)
            result_df = result_df.reindex(df.index)

            # Overwrite existing columns (or create new ones) with LLM output
            for col in result_df.columns:
                df[col] = result_df[col]

        output_df = df

        # Save output
        logger.info("Saving results to %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info("✓ Saved %d rows", len(output_df))

        # Print summary
        separator = "=" * 56
        logger.info(separator)

        duration = time() - start_time
        logger.info(
            "Summary | total=%d processed=%d skipped=%d success=%d errors=%d | duration=%.1fs",
            len(df),
            processed,
            skipped,
            processed - len(errors),
            len(errors),
            duration,
        )

        if errors:
            first_errors = "; ".join(errors[:5])
            logger.warning("Errors (%d): %s", len(errors), first_errors)
            if len(errors) > 5:
                logger.warning("... %d additional error(s)", len(errors) - 5)

        logger.info(separator)

    def _retry_failed_rows(
        self,
        failed_rows,
        results: List[dict],
        columns: List[str],
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
    ):
        """Retry rows that failed during the initial pass.

        Args:
            failed_rows: List of failure metadata dicts.
            results: List of per-row outputs to update.
            columns: Columns to read from the input row.
            prompt: Prompt to send to the LLM.
            response_model: Optional response schema.

        Returns:
            List of failure metadata that still failed after retry.
        """
        logger.info(
            "Retrying %d failed row(s) before finalizing output...",
            len(failed_rows),
        )
        remaining_failures = []

        for failure in failed_rows:
            row = failure["row"]
            row_num = failure["row_num"]
            position = failure["position"]

            try:
                input_data = {col: row[col] for col in columns}
                output = self.llm_client.generate_structured_output(
                    prompt, input_data, response_model
                )
                results[position] = output
                logger.info(color_text(f"Row {row_num} ✓ recovered on retry", "green"))
            except Exception as retry_error:
                formatted = format_error_message(retry_error)
                failure["error"] = formatted
                remaining_failures.append(failure)
                logger.error(
                    color_text(f"Row {row_num} ✗ retry failed: {formatted}", "red")
                )

        if remaining_failures:
            logger.warning(
                "Unable to recover %d row(s) after fallback.", len(remaining_failures)
            )
        else:
            logger.info("Recovered all failed rows successfully.")

        return remaining_failures

    def preview_sample(
        self,
        input_path: Path | str,
        columns: List[str],
        prompt: str,
        num_rows: int = DEFAULT_PREVIEW_ROWS,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> None:
        """Preview LLM output on a sample of rows without saving.

        Args:
            input_path: Path to the input CSV file.
            columns: List of column names to use as input for LLM.
            prompt: User-provided prompt describing the task.
            num_rows: Number of rows to preview (default: 3).
            response_model: Optional Pydantic model for structured output.
        """
        input_path = Path(input_path)

        if response_model is None:
            raise ValueError(
                "Output schema is required. Dynamic schema is not supported."
            )

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load CSV
        df = pd.read_csv(input_path)

        # Validate columns
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Columns not found in CSV: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Process sample rows
        logger.info(
            "Preview mode | rows=%d | columns=%s",
            num_rows,
            ",".join(columns),
        )
        sample_df = df.head(num_rows)
        total_rows = len(sample_df)

        def _truncate(value):
            if isinstance(value, str) and len(value) > 100:
                return value[:100] + "..."
            return value

        for idx, row in sample_df.iterrows():
            row_num = idx + 1
            prefix = f"[preview {row_num}/{total_rows}]"
            input_preview = {}
            for col in columns:
                value = row[col]
                input_preview[col] = _truncate(value)
            logger.info("%s input=%s", prefix, input_preview)

            try:
                # Extract input data
                input_data = {col: row[col] for col in columns}

                # Generate output
                output = self.llm_client.generate_structured_output(
                    prompt, input_data, response_model
                )

                logger.info(
                    "%s output=%s",
                    prefix,
                    json.dumps(output, ensure_ascii=False),
                )

            except Exception as e:
                logger.error("%s ✗ %s", prefix, format_error_message(e))
