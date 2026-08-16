"""GenAI explanation utilities.

Safety contract: this layer provides decision support only. It must never
invent insurance coverage, policy terms, payout commitments, legal
conclusions, or medical diagnoses.
"""

SYSTEM_PROMPT = (
    'You explain insurance severity predictions for analysts. Be factual, '
    'concise, uncertainty-aware, and never invent policy coverage, legal '
    'conclusions, diagnoses, or payout commitments.'
)


def risk_tier(prediction: float) -> str:
    """Bucket a predicted severity value into a coarse risk tier."""
    if prediction < 500:
        return 'LOW'
    if prediction < 2000:
        return 'MEDIUM'
    if prediction < 5000:
        return 'HIGH'
    return 'CRITICAL'


def build_explanation_prompt(claim, prediction, top_factors=None):
    """Build a structured prompt for an LLM to explain a severity prediction."""
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Claim features: {claim}\n"
        f"Predicted severity: {prediction:.2f}\n"
        f"Top model factors: {top_factors or 'not available'}\n"
        "Explain the result, assumptions, uncertainty, and recommended human review checks."
    )


def deterministic_explanation(prediction, top_factors=None):
    """Rule-based explanation that does not require calling an LLM API."""
    factors = ', '.join(top_factors) if top_factors else 'the supplied claim features'
    tier = risk_tier(prediction)
    return (
        f"Estimated claim severity is {prediction:.2f} (risk tier: {tier}). "
        f"The estimate is associated with {factors}. "
        "Validate coverage, data quality, and uncertainty with a claims professional."
    )
