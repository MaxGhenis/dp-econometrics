"""
Privacy budget accounting for differential privacy.

Tracks cumulative privacy loss across multiple queries using
composition theorems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Literal
import numpy as np


@dataclass
class QueryRecord:
    """Record of a single privacy-consuming query."""
    epsilon: float
    delta: float
    timestamp: datetime
    query_name: Optional[str] = None
    query_type: Optional[str] = None


class PrivacyAccountant:
    """
    Track privacy budget across multiple queries.

    This class implements privacy composition to track cumulative
    privacy loss as queries are executed. It supports both basic
    sequential composition and advanced Rényi DP composition.

    Parameters
    ----------
    epsilon_budget : float
        Total epsilon budget available.
    delta_budget : float
        Total delta budget available.
    composition : {'basic', 'rdp'}
        Composition method. 'basic' uses simple sequential composition,
        'rdp' uses Rényi DP for tighter bounds.

    Attributes
    ----------
    epsilon_spent : float
        Total epsilon consumed so far.
    delta_spent : float
        Total delta consumed so far.
    queries : int
        Number of queries executed.

    Examples
    --------
    >>> accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)
    >>> accountant.spend(epsilon=0.3, delta=1e-6)
    >>> print(f"Remaining: {accountant.epsilon_remaining}")
    Remaining: 0.7

    References
    ----------
    Dwork, C., Rothblum, G. N., & Vadhan, S. (2010). Boosting and
    differential privacy. In FOCS 2010.

    Mironov, I. (2017). Rényi differential privacy. In CSF 2017.
    """

    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float,
        composition: Literal["basic", "rdp"] = "basic"
    ):
        if epsilon_budget <= 0:
            raise ValueError("epsilon_budget must be positive")
        if delta_budget <= 0:
            raise ValueError("delta_budget must be positive")

        self.epsilon_budget = epsilon_budget
        self.delta_budget = delta_budget
        self.composition = composition

        self._epsilon_spent = 0.0
        self._delta_spent = 0.0
        self._queries: List[QueryRecord] = []

        # For RDP composition
        if composition == "rdp":
            self._rdp_orders = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]
            self._rdp_spent = np.zeros(len(self._rdp_orders))

    @property
    def epsilon_spent(self) -> float:
        """Total epsilon consumed."""
        if self.composition == "rdp":
            return self._rdp_to_dp_epsilon()
        return self._epsilon_spent

    @property
    def delta_spent(self) -> float:
        """Total delta consumed."""
        return self._delta_spent

    @property
    def epsilon_remaining(self) -> float:
        """Remaining epsilon budget."""
        return max(0, self.epsilon_budget - self.epsilon_spent)

    @property
    def delta_remaining(self) -> float:
        """Remaining delta budget."""
        return max(0, self.delta_budget - self.delta_spent)

    @property
    def queries(self) -> int:
        """Number of queries executed."""
        return len(self._queries)

    def _rdp_to_dp_epsilon(self) -> float:
        """Convert RDP accounting to (ε,δ)-DP."""
        if not hasattr(self, '_rdp_spent') or np.sum(self._rdp_spent) == 0:
            return 0.0

        # Convert RDP to (ε,δ)-DP using optimal order
        epsilons = []
        for i, alpha in enumerate(self._rdp_orders):
            if alpha <= 1:  # pragma: no cover
                continue
            eps = self._rdp_spent[i] + np.log(1 / self.delta_budget) / (alpha - 1)
            epsilons.append(eps)

        return min(epsilons) if epsilons else self._epsilon_spent

    def _add_rdp(self, epsilon: float, delta: float) -> None:
        """Add to RDP accounting."""
        # Simplified RDP: for Gaussian mechanism with σ = sensitivity * sqrt(2ln(1.25/δ))/ε
        # RDP at order α is approximately α / (2σ²)
        # For now, use basic composition for RDP (can be improved)
        for i, alpha in enumerate(self._rdp_orders):
            # This is a simplified approximation
            # A proper implementation would use the exact RDP formula
            self._rdp_spent[i] += epsilon * alpha / 2

    def can_afford(self, epsilon: float, delta: float = 0.0) -> bool:
        """
        Check if a query with given cost is affordable.

        Parameters
        ----------
        epsilon : float
            Epsilon cost of the query.
        delta : float
            Delta cost of the query.

        Returns
        -------
        bool
            True if the query can be afforded.
        """
        projected_eps = self.epsilon_spent + epsilon
        projected_delta = self.delta_spent + delta

        return (
            projected_eps <= self.epsilon_budget and
            projected_delta <= self.delta_budget
        )

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        query_name: Optional[str] = None,
        query_type: Optional[str] = None
    ) -> None:
        """
        Record privacy expenditure for a query.

        Parameters
        ----------
        epsilon : float
            Epsilon cost of the query.
        delta : float
            Delta cost of the query.
        query_name : str, optional
            Name/identifier for the query.
        query_type : str, optional
            Type of query (e.g., 'ols', 'logit').

        Raises
        ------
        ValueError
            If the query would exceed the privacy budget.
        """
        if not self.can_afford(epsilon, delta):
            raise ValueError(
                f"Privacy budget exceeded. "
                f"Requested ε={epsilon}, δ={delta}. "
                f"Available ε={self.epsilon_remaining:.4f}, "
                f"δ={self.delta_remaining:.2e}"
            )

        # Record the query
        record = QueryRecord(
            epsilon=epsilon,
            delta=delta,
            timestamp=datetime.now(),
            query_name=query_name,
            query_type=query_type
        )
        self._queries.append(record)

        # Update accounting
        if self.composition == "basic":
            self._epsilon_spent += epsilon
        elif self.composition == "rdp":  # pragma: no branch
            self._add_rdp(epsilon, delta)

        self._delta_spent += delta

    def get_history(self) -> List[dict]:
        """
        Get history of all queries.

        Returns
        -------
        list of dict
            Each dict contains query details: epsilon, delta,
            timestamp, query_name, query_type.
        """
        return [
            {
                "epsilon": q.epsilon,
                "delta": q.delta,
                "timestamp": q.timestamp,
                "query_name": q.query_name,
                "query_type": q.query_type,
            }
            for q in self._queries
        ]

    def summary(self) -> str:
        """Get a summary of privacy budget status."""
        return (
            f"Privacy Budget Summary\n"
            f"======================\n"
            f"Budget:    ε = {self.epsilon_budget:.4f}, "
            f"δ = {self.delta_budget:.2e}\n"
            f"Spent:     ε = {self.epsilon_spent:.4f}, "
            f"δ = {self.delta_spent:.2e}\n"
            f"Remaining: ε = {self.epsilon_remaining:.4f}, "
            f"δ = {self.delta_remaining:.2e}\n"
            f"Queries:   {self.queries}\n"
            f"Method:    {self.composition} composition"
        )

    def __repr__(self) -> str:
        return (
            f"PrivacyAccountant(budget=({self.epsilon_budget}, {self.delta_budget}), "
            f"spent=({self.epsilon_spent:.4f}, {self.delta_spent:.2e}), "
            f"queries={self.queries})"
        )
