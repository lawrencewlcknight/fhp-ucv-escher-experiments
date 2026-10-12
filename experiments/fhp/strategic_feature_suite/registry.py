"""Standard-library-only registry, also imported by lightweight cloud controllers."""
BASELINE_RUN_ID = "exp10-features-20261001-161740"
BASELINE_REF = "e66d4da515eb212e5026a965ac5c39c86144c901"
SPECS = {
    29: ("relative_strength", "A", "Overall relative strength"),
    30: ("opponent_classes", "AB", "Relative strength and opponent classes"),
    31: ("blocker_features", "AC", "Relative strength and blockers"),
    32: ("preflop_distribution", "D", "Preflop strength distribution"),
    33: ("strength_betting", "AE", "Relative strength and betting interactions"),
}
