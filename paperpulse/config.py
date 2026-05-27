"""
PaperPulse configuration.

Central place for user-facing settings that affect paper selection.
Edit INTEREST_PROFILE to tune which papers get selected for you.
"""

# ---------------------------------------------------------------------------
# Interest profile
# ---------------------------------------------------------------------------
# Describes the ideal reader of PaperPulse.
# Used by both the embedding pre-filter and the LLM scoring step.
# Edit this to adjust which papers get selected for you.

INTEREST_PROFILE = """
AI clinical decision support
Tumor board decision support AI
AI-assisted diagnosis
Clinical outcome prediction
NLP applied to clinical notes
NLP for medical documentation and discharge summaries
Information extraction from clinical text
Electronic health record systems and EHR integration
Health data interoperability standards
FHIR implementation and clinical data exchange
Digital therapeutics
Real-world clinical AI applications
Novel technical approaches enabling clinical AI
"""