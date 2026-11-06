"""radar-wow verification: matching radar estimates to references (gauges, MRMS) and scoring them."""

from .models import hrrr_field, hrrr_freezing_level
from .mrms import PRODUCTS, fetch_mrms, gate_latlon, scores, to_mrms_grid

PLANNED = ["match_gauges", "stratified_scores"]
__all__ = ["hrrr_field", "hrrr_freezing_level", "PRODUCTS", "fetch_mrms", "gate_latlon", "scores", "to_mrms_grid"]
