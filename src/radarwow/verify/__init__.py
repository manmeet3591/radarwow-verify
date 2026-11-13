"""radar-wow verification: matching radar estimates to references (gauges, MRMS) and scoring them."""

from .gauges import basic_qc, fetch_fmi, fmi_accumulation_at, fetch_hads, gross_error_qc, hads_stations
from .models import hrrr_field, hrrr_freezing_level, t2m_freezing_level
from .mrms import PRODUCTS, fetch_mrms, gate_latlon, scores, to_mrms_grid

PLANNED = ["stratified_scores"]
__all__ = ["t2m_freezing_level", "fetch_fmi", "fmi_accumulation_at", "basic_qc", "fetch_hads", "gross_error_qc", "hads_stations", "hrrr_field", "hrrr_freezing_level", "PRODUCTS", "fetch_mrms", "gate_latlon", "scores", "to_mrms_grid"]
