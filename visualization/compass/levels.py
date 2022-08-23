from dataclasses import dataclass


@dataclass
class InnerLevel:
    """Inner CLEVA-Compass level with method attributes."""

    
    Risk_Control: int
    Proftability: int
    Explainability: int
    Reliability: int
    Diversity: int
    University: int
    

    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        blank.tex file.
        """
        for item in [
            self.Risk_Control,
            self.Proftability,
            self.Explainability,
            self.Reliability,
            self.Diversity,
            self.University,
        ]:
            yield item


@dataclass
class OuterLevel:
    """Outer CLEVA-Compass level with measurement attributes."""
    country: bool
    assert_type: bool
    time_scale: bool
    risk: bool
    risk_adjusted: bool
    extreme_market: bool
    profit: bool
    alpha_decay: bool

    equity_curve: bool
    profile: bool
    variability: bool
    rank_order: bool
    t_SNE: bool
    entropy: bool
    correlation: bool
    rolling_window: bool

    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        cleva_template.tex file.
        """
        for item in [
            
            
            self.country,
            self.assert_type,
            self.time_scale,
            self.risk,
            self.risk_adjusted,
            self.extreme_market,
            self.profit,
            self.alpha_decay,

            self.equity_curve,
            self.profile,
            self.variability,
            self.rank_order,
            self.t_SNE,
            self.entropy,
            self.correlation,
            self.rolling_window,
            



        ]:
            yield item


@dataclass
class CompassEntry:
    """Compass entry containing color, label, and attributes."""

    color: str  # Color, can be one of [magenta, green, blue, orange, cyan, brown]
    label: str  # Legend label
    inner_level: InnerLevel
    outer_level: OuterLevel
