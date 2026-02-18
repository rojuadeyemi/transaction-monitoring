import pandas as pd


class RiskScoringEngine:
    def __init__(self, weights_df: pd.DataFrame):

        self.weights_df = weights_df.copy()

        # Convert weights table into dictionary
        self.weights_dict = (
            self.weights_df
            .set_index("Flag Name")["Weight"]
            .to_dict()
        )

        # Maximum possible score (for normalization)
        self.max_score = self.weights_df["Weight"].sum()

    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:

        flag_columns = self.weights_dict.keys()

        # Ensure only available columns are used
        available_flags = [col for col in flag_columns if col in df.columns]

        # Normalize to percentage
        df["risk_score"] = (sum(
            df[col] * self.weights_dict[col]
            for col in available_flags
        ) / self.max_score) * 100

        # Assign risk level
        df["risk_level"] = df["risk_score"].apply(self.assign_risk_level)

        return df

    @staticmethod
    def assign_risk_level(score):
        if score >=30:
            return "Critical"
        elif score >= 10:
            return "High"
        elif score >= 5:
            return "Medium"
        else:
            return "Low"
