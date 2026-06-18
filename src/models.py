from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SECFilingDetails(BaseModel):
    company_name: str = Field(
        description="The formal, official name of the corporation or entity filing the report. Output in all CAPS."
    )
    ticker: str = Field(
        description="The official stock exchange trading ticker symbol. Output in all CAPS."
    )
    fiscal_year: int = Field(
        ge=1934,
        le=2030,
        description="The specific four-digit calendar year for which this report is covering."
    )
    filing_type: Literal["10-K", "10-Q"] = Field(
        description="The standardized regulatory SEC form type or submission designation."
    )
    period_ended: Optional[date] = Field(
        default=None,
        description="The exact balance sheet or reporting period end date (Format: YYYY-MM-DD). If not present, use null."
    )
    filing_date: date = Field(
        description="The official date this document was submitted or published to the SEC EDGAR system (Format: YYYY-MM-DD)."
    )