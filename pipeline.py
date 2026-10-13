"""
Data loading, time reconstruction, feature engineering and the time-based split.

Shared by train.py (which fits everything) and app.py (which re-uses the same
loading / feature code for the distribution plots and to score a new client
with the *saved* preprocessor). Nothing in here trains a model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
ARTIFACTS = ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Business assumptions (see REPORT.md, "Business framing"). Hypothetical but
# explicit, so the decision threshold is derived from them rather than from
# an arbitrary 0.5.
# ---------------------------------------------------------------------------
# Cost of putting ONE CLIENT on the call list. A client takes ~2.6 call
# attempts on average (mean of `campaign` in the training months); each attempt
# is ~8 min of agent time incl. dialing, no-answers and wrap-up at ~EUR 30/h
# fully loaded (EUR 4), so ~EUR 10 per client, plus ~EUR 2 allowance for the
# annoyance / complaint / opt-out risk of an unwanted call.
CALL_COST = 12.0
# Value of one new term deposit: an average ~EUR 4,000 deposit earning ~1% net
# interest margin over a 1-year term.
DEPOSIT_VALUE = 40.0
# If predicted probabilities were perfectly calibrated, calling a client is
# worth it exactly when p * DEPOSIT_VALUE > CALL_COST, i.e. p > 0.30.
BREAK_EVEN_P = CALL_COST / DEPOSIT_VALUE

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Split boundaries on the reconstructed calendar (inclusive start dates).
# Train: May 2008 - Dec 2009 | Validation: Mar - Jul 2010 | Test: Aug - Nov 2010
VAL_START = pd.Timestamp("2010-03-01")
TEST_START = pd.Timestamp("2010-08-01")


def load_raw() -> pd.DataFrame:
    """Load the UCI file and attach a reconstructed calendar month.

    The file has a `month` column but no year. Rows are stored in
    chronological order (May 2008 -> Nov 2010, documented by the dataset
    authors), so every time the month name changes from one row to the next
    we have moved to the next calendar month of the campaign. Counting those
    changes gives the year back.
    """
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df["y"] = (df["y"] == "yes").astype(int)

    month_num = df["month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    new_block = month_num != month_num.shift()
    # A new year starts whenever the month number goes *down* (dec -> mar).
    year = 2008 + (new_block & (month_num < month_num.shift())).cumsum()
    df["period"] = pd.to_datetime(dict(year=year, month=month_num, day=1))
    df["row_order"] = np.arange(len(df))
    return df


def clean(df: pd.DataFrame, keep_duration: bool = False) -> pd.DataFrame:
    """Data-quality decisions that apply before any feature is built."""
    df = df.copy()
    # 12 rows are exact duplicates *including* call duration to the second.
    # Two different clients with identical everything is implausible, so treat
    # them as logging duplicates.
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_order"])
    # `duration` is only known after the call has ended, and duration == 0
    # literally means y == 0. Using it would be target leakage for a model
    # that decides *whether* to call. Dropped for good.
    if not keep_duration:   # (True only for the leakage demo in the ablation)
        df = df.drop(columns=["duration"])
    return df.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready columns from the cleaned raw data (row-wise only,
    so it cannot leak information across the split)."""
    out = pd.DataFrame(index=df.index)

    # --- previous-campaign history -----------------------------------------
    # pdays == 999 is a sentinel for "never contacted before" (96% of rows),
    # not a real number of days. Feeding it raw would tell a linear model that
    # never-contacted clients are "very far in the past", dominating the scale.
    contacted = df["pdays"] != 999
    out["prev_contacted"] = contacted.astype(int)
    # For those who were contacted, recency matters: a contact < 1 week ago is
    # still warm. Expressed as an indicator rather than raw days because
    # pdays only takes values 0..27 for the 4% of contacted rows.
    out["prev_recent_week"] = (contacted & (df["pdays"] <= 7)).astype(int)
    # Previous-contact counts are tiny integers with a long thin tail.
    out["previous_clip"] = df["previous"].clip(upper=3)
    # poutcome is one-hot encoded below; 'nonexistent' == not prev_contacted
    # (mostly), so it is the reference level.

    # --- this campaign ------------------------------------------------------
    # Each row is a client's LAST contact in the campaign. `campaign` (number
    # of attempts so far), `day_of_week` and `month` describe that last call,
    # so they are only known once calling has ended -- and clients who say yes
    # stop being called, so `campaign` partly encodes the answer. The decision
    # here is made *before* calling (who goes on the list), so they are not
    # used. What IS known up front: which phone number we hold.
    out["contact_cellular"] = (df["contact"] == "cellular").astype(int)

    # --- client profile -----------------------------------------------------
    # The age effect is U-shaped (students and retirees convert most), which a
    # single linear age coefficient cannot represent. Bins let the linear
    # model learn a separate level per life stage.
    out["age_band"] = pd.cut(
        df["age"], bins=[0, 25, 30, 40, 50, 60, 120],
        labels=["<=25", "26-30", "31-40", "41-50", "51-60", "60+"],
    ).astype(str)
    # 'default' has only 3 'yes' rows; the informative split is
    # "declared no default" vs "refused to say" (unknown converts at 5% vs 13%).
    out["default_unknown"] = (df["default"] != "no").astype(int)
    out["housing_loan"] = (df["housing"] == "yes").astype(int)
    out["personal_loan"] = (df["loan"] == "yes").astype(int)
    # 18 'illiterate' rows cannot support their own coefficient: merge with
    # the closest level.
    out["education"] = df["education"].replace({"illiterate": "basic.4y"})
    out["job"] = df["job"]
    out["marital"] = df["marital"].replace({"unknown": "single"})
    out["poutcome"] = df["poutcome"]

    # --- macro-economic context --------------------------------------------
    # Monthly/quarterly indicators that are public at call time. euribor3m,
    # emp_var_rate and nr_employed are ~0.95 correlated (all track the 2008-09
    # crisis), so keep one rate (euribor3m) plus the two indicators that carry
    # different information (consumer prices, consumer confidence).
    out["euribor3m"] = df["euribor3m"]
    out["cons_price_idx"] = df["cons_price_idx"]
    out["cons_conf_idx"] = df["cons_conf_idx"]

    return out


NUMERIC = ["previous_clip", "euribor3m", "cons_price_idx", "cons_conf_idx"]
BINARY = ["prev_contacted", "prev_recent_week", "contact_cellular",
          "default_unknown", "housing_loan", "personal_loan"]
CATEGORICAL = ["age_band", "education", "job", "marital", "poutcome"]
# Reference levels dropped from each one-hot block, so the intercept stays
# interpretable and the design matrix is not perfectly collinear.
REFERENCE = {"age_band": "31-40", "education": "high.school", "job": "admin.",
             "marital": "married", "poutcome": "nonexistent"}


def make_preprocessor() -> ColumnTransformer:
    """Scaling + one-hot encoding, to be *fit on the training split only*.

    Standardizing numeric columns matters twice: L2 regularization penalizes
    all coefficients equally, so they must be on a comparable scale, and
    gradient descent in the two PyTorch versions converges far faster on a
    well-conditioned problem.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", BINARY),
            ("cat", OneHotEncoder(drop=[REFERENCE[c] for c in CATEGORICAL],
                                  handle_unknown="ignore", sparse_output=False),
             CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def time_split(df: pd.DataFrame):
    """Return boolean masks for train / validation / test by calendar month."""
    train = df["period"] < VAL_START
    val = (df["period"] >= VAL_START) & (df["period"] < TEST_START)
    test = df["period"] >= TEST_START
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


def profit(y_true: np.ndarray, call: np.ndarray,
           call_cost: float = CALL_COST, deposit_value: float = DEPOSIT_VALUE) -> float:
    """Campaign profit (EUR) of calling exactly the clients where call == 1."""
    call = call.astype(bool)
    return float(deposit_value * y_true[call].sum() - call_cost * call.sum())
