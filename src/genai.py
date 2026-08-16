SYSTEM_PROMPT='You explain insurance severity predictions for analysts. Be factual, concise, uncertainty-aware, and never invent policy coverage, legal conclusions, diagnoses, or payout commitments.'

def build_explanation_prompt(claim, prediction, top_factors=None):
    return f'''{SYSTEM_PROMPT}\nClaim features: {claim}\nPredicted severity: {prediction:.2f}\nTop model factors: {top_factors or 'not available'}\nExplain the result, assumptions, uncertainty, and recommended human review checks.'''

def deterministic_explanation(prediction, top_factors=None):
    factors=', '.join(top_factors or []) or 'the supplied claim features'
    return f'Estimated claim severity is {prediction:.2f}. The estimate is associated with {factors}. Validate coverage, data quality, and uncertainty with a claims professional.'
