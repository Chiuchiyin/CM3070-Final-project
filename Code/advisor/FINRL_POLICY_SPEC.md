# Selected FinRL Policy Specification

The production RL path follows the portfolio-allocation notebook
`FinRL/FinRL_PortfolioAllocation_NeurIPS_2020.ipynb`, with A2C selected as the
single algorithm. The production application will not execute notebook cells.

- Universe: the configured five MVP tickers, sorted and versioned.
- Action: one logit per asset plus cash, transformed by stable softmax.
- Constraint: finite long-only weights summing to one.
- Observation: trailing close returns, current portfolio weights, and cash.
- Reward: net log portfolio growth after transaction costs and slippage.
- Splits: chronological train, validation, and untouched test intervals.
- Training: offline only; the Shiny request path performs inference only.
- Artifact: saved policy plus metadata containing source, contract, seed, data
  digest, and split boundaries.

The multi-stock notebook variants are not the production path because their
share-transaction action spaces do not match the MVP target-weight contract.
