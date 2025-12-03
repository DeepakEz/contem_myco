#!/usr/bin/env python3
"""
Society Environment v0 - Socio-Economic Simulation
===================================================
Simulates a society of 100 citizens with:
- Income/wealth distribution
- Basic needs (food, housing, health)
- Social trust metrics
- Crime and suffering
- Policy interventions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CitizenClass(Enum):
    """Socio-economic classes"""
    POOR = 0
    LOWER_MIDDLE = 1
    MIDDLE = 2
    UPPER_MIDDLE = 3
    WEALTHY = 4


@dataclass
class CitizenState:
    """State of a citizen"""
    citizen_id: int
    wealth: float = 100.0  # Starting wealth
    income: float = 10.0   # Income per step
    happiness: float = 0.5  # 0.0 to 1.0
    trust: float = 0.5  # Trust in society
    health: float = 1.0
    has_housing: bool = True
    has_food: bool = True
    suffering_level: float = 0.0
    committed_crime: bool = False
    crime_count: int = 0


class PolicyType(Enum):
    """Types of policy interventions"""
    NO_POLICY = "no_policy"
    UNIVERSAL_BASIC_INCOME = "ubi"
    PROGRESSIVE_TAX = "progressive_tax"
    PUBLIC_HEALTHCARE = "public_healthcare"
    EDUCATION_SUBSIDY = "education_subsidy"
    HOUSING_ASSISTANCE = "housing_assistance"
    WELFARE_SUPPORT = "welfare"
    COMMUNITY_PROGRAMS = "community_programs"


class SocietyEnv:
    """
    Society simulation environment

    Simulates 100 citizens with socio-economic dynamics and policy interventions
    """

    def __init__(
        self,
        num_citizens: int = 100,
        initial_inequality: float = 0.4,  # Gini coefficient
        base_crime_rate: float = 0.05,
        seed: Optional[int] = None
    ):
        self.num_citizens = num_citizens
        self.initial_inequality = initial_inequality
        self.base_crime_rate = base_crime_rate

        if seed is not None:
            np.random.seed(seed)

        # State
        self.step_count = 0
        self.citizens: Dict[int, CitizenState] = {}

        # Economic parameters
        self.cost_of_living = 8.0  # Per step
        self.housing_cost = 15.0
        self.healthcare_cost = 10.0
        self.food_cost = 5.0

        # Policy state
        self.active_policies: List[PolicyType] = []
        self.policy_budget = 0.0
        self.total_tax_collected = 0.0

        # Statistics
        self.total_crimes = 0
        self.total_suffering = 0.0
        self.inequality_history = []
        self.trust_history = []

        # Initialize citizens
        self._initialize_citizens()

    def _initialize_citizens(self):
        """Initialize citizens with realistic wealth distribution"""
        # Generate wealth distribution with specified inequality
        # Using Pareto distribution for realistic inequality

        alpha = 1.0 / (self.initial_inequality + 0.1)  # Shape parameter

        for citizen_id in range(self.num_citizens):
            # Wealth from Pareto distribution
            base_wealth = np.random.pareto(alpha) * 50 + 50

            # Income correlated with wealth
            base_income = 5 + (base_wealth / 20)

            # Add some randomness
            base_income += np.random.normal(0, 2)

            self.citizens[citizen_id] = CitizenState(
                citizen_id=citizen_id,
                wealth=base_wealth,
                income=max(1.0, base_income),
                happiness=0.5 + np.random.normal(0, 0.1),
                trust=0.5 + np.random.normal(0, 0.1),
                health=0.9 + np.random.normal(0, 0.05)
            )

    def step(
        self,
        policy_action: Optional[PolicyType] = None
    ) -> Tuple[Dict[str, float], float, bool, Dict]:
        """
        Step the society simulation

        Args:
            policy_action: Optional policy to implement

        Returns:
            observation, reward, done, info
        """
        self.step_count += 1

        # Apply policy if provided
        if policy_action and policy_action != PolicyType.NO_POLICY:
            self._apply_policy(policy_action)

        # Economic cycle
        self._economic_cycle()

        # Social dynamics
        self._social_dynamics()

        # Update statistics
        self._update_statistics()

        # Calculate reward (negative suffering + positive trust)
        reward = self._calculate_reward()

        # Generate observation
        observation = self._get_observation()

        # Info
        info = self.get_statistics()

        # Done if simulation limit reached
        done = self.step_count >= 200

        return observation, reward, done, info

    def _apply_policy(self, policy: PolicyType):
        """Apply policy intervention"""
        self.active_policies.append(policy)

        if policy == PolicyType.UNIVERSAL_BASIC_INCOME:
            # Give everyone base income
            ubi_amount = 5.0
            for citizen in self.citizens.values():
                citizen.wealth += ubi_amount
                citizen.happiness += 0.02
                citizen.trust += 0.01
            self.policy_budget -= ubi_amount * self.num_citizens

        elif policy == PolicyType.PROGRESSIVE_TAX:
            # Tax wealthy more
            for citizen in self.citizens.values():
                if citizen.wealth > 150:
                    tax = citizen.income * 0.3
                elif citizen.wealth > 100:
                    tax = citizen.income * 0.2
                else:
                    tax = citizen.income * 0.1

                citizen.wealth -= tax
                self.total_tax_collected += tax
                self.policy_budget += tax

        elif policy == PolicyType.PUBLIC_HEALTHCARE:
            # Reduce healthcare costs
            for citizen in self.citizens.values():
                if citizen.health < 0.8:
                    citizen.health = min(1.0, citizen.health + 0.1)
                    citizen.suffering_level = max(0.0, citizen.suffering_level - 0.1)
            self.policy_budget -= 10.0 * self.num_citizens

        elif policy == PolicyType.HOUSING_ASSISTANCE:
            # Help with housing
            for citizen in self.citizens.values():
                if not citizen.has_housing:
                    if np.random.random() < 0.3:  # 30% chance to get housing
                        citizen.has_housing = True
                        citizen.happiness += 0.15
                        citizen.trust += 0.1
            self.policy_budget -= 5.0 * self.num_citizens

        elif policy == PolicyType.WELFARE_SUPPORT:
            # Support poorest citizens
            poor_citizens = sorted(
                self.citizens.values(),
                key=lambda c: c.wealth
            )[:self.num_citizens // 5]  # Bottom 20%

            for citizen in poor_citizens:
                citizen.wealth += 8.0
                citizen.happiness += 0.05
                citizen.suffering_level = max(0.0, citizen.suffering_level - 0.1)
            self.policy_budget -= 8.0 * len(poor_citizens)

        elif policy == PolicyType.COMMUNITY_PROGRAMS:
            # Increase trust and reduce crime
            for citizen in self.citizens.values():
                citizen.trust = min(1.0, citizen.trust + 0.03)
                citizen.happiness += 0.02
            self.policy_budget -= 3.0 * self.num_citizens

    def _economic_cycle(self):
        """Process economic activities"""
        for citizen in self.citizens.values():
            # Earn income
            citizen.wealth += citizen.income

            # Pay for necessities
            if citizen.wealth >= self.food_cost:
                citizen.wealth -= self.food_cost
                citizen.has_food = True
            else:
                citizen.has_food = False
                citizen.health -= 0.05

            if citizen.wealth >= self.housing_cost:
                citizen.wealth -= self.housing_cost
                citizen.has_housing = True
            else:
                citizen.has_housing = False
                citizen.happiness -= 0.1

            # Healthcare (optional but important)
            if citizen.health < 0.7 and citizen.wealth >= self.healthcare_cost:
                citizen.wealth -= self.healthcare_cost
                citizen.health = min(1.0, citizen.health + 0.15)

            # Wealth cannot go negative
            citizen.wealth = max(0.0, citizen.wealth)

    def _social_dynamics(self):
        """Process social dynamics: trust, crime, suffering"""
        for citizen in self.citizens.values():
            # Update happiness based on needs
            if citizen.has_food and citizen.has_housing and citizen.health > 0.7:
                citizen.happiness = min(1.0, citizen.happiness + 0.02)
            else:
                citizen.happiness = max(0.0, citizen.happiness - 0.05)

            # Update trust based on happiness and equality
            avg_wealth = np.mean([c.wealth for c in self.citizens.values()])
            wealth_ratio = citizen.wealth / (avg_wealth + 1.0)

            if citizen.happiness > 0.6:
                citizen.trust = min(1.0, citizen.trust + 0.01)
            elif citizen.happiness < 0.3:
                citizen.trust = max(0.0, citizen.trust - 0.02)

            # Suffering from unmet needs
            citizen.suffering_level = 0.0

            if not citizen.has_food:
                citizen.suffering_level += 0.3
            if not citizen.has_housing:
                citizen.suffering_level += 0.2
            if citizen.health < 0.5:
                citizen.suffering_level += 0.2
            if citizen.happiness < 0.3:
                citizen.suffering_level += 0.15

            citizen.suffering_level = min(1.0, citizen.suffering_level)

            # Crime likelihood based on desperation
            crime_probability = self.base_crime_rate

            if citizen.wealth < 20:
                crime_probability += 0.1
            if citizen.suffering_level > 0.5:
                crime_probability += 0.1
            if citizen.trust < 0.3:
                crime_probability += 0.05

            if np.random.random() < crime_probability:
                citizen.committed_crime = True
                citizen.crime_count += 1
                self.total_crimes += 1

                # Crime reduces trust in others
                for other in self.citizens.values():
                    other.trust = max(0.0, other.trust - 0.01)

            else:
                citizen.committed_crime = False

    def _update_statistics(self):
        """Update historical statistics"""
        # Gini coefficient
        wealths = sorted([c.wealth for c in self.citizens.values()])
        n = len(wealths)

        if n > 0 and sum(wealths) > 0:
            cumsum = np.cumsum(wealths)
            gini = (2 * np.sum((np.arange(n) + 1) * wealths)) / (n * np.sum(wealths)) - (n + 1) / n
        else:
            gini = 0.0

        self.inequality_history.append(gini)

        # Average trust
        avg_trust = np.mean([c.trust for c in self.citizens.values()])
        self.trust_history.append(avg_trust)

    def _calculate_reward(self) -> float:
        """Calculate reward for policy agent"""
        # Goal: minimize suffering, crime, inequality
        #       maximize trust, happiness

        avg_suffering = np.mean([c.suffering_level for c in self.citizens.values()])
        avg_trust = np.mean([c.trust for c in self.citizens.values()])
        avg_happiness = np.mean([c.happiness for c in self.citizens.values()])
        crime_rate = self.total_crimes / self.num_citizens if self.step_count > 0 else 0.0
        inequality = self.inequality_history[-1] if self.inequality_history else 0.5

        reward = (
            -avg_suffering * 2.0 +
            avg_trust * 1.0 +
            avg_happiness * 1.0 -
            crime_rate * 5.0 -
            inequality * 1.0
        )

        return reward

    def _get_observation(self) -> Dict[str, float]:
        """Generate observation"""
        wealths = [c.wealth for c in self.citizens.values()]

        return {
            'avg_wealth': np.mean(wealths),
            'median_wealth': np.median(wealths),
            'wealth_std': np.std(wealths),
            'inequality': self.inequality_history[-1] if self.inequality_history else 0.5,
            'avg_happiness': np.mean([c.happiness for c in self.citizens.values()]),
            'avg_trust': np.mean([c.trust for c in self.citizens.values()]),
            'avg_health': np.mean([c.health for c in self.citizens.values()]),
            'avg_suffering': np.mean([c.suffering_level for c in self.citizens.values()]),
            'crime_rate': self.total_crimes / (self.step_count * self.num_citizens) if self.step_count > 0 else 0.0,
            'homeless_rate': sum(1 for c in self.citizens.values() if not c.has_housing) / self.num_citizens,
            'hungry_rate': sum(1 for c in self.citizens.values() if not c.has_food) / self.num_citizens
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        obs = self._get_observation()
        obs.update({
            'step': self.step_count,
            'total_crimes': self.total_crimes,
            'active_policies': [p.value for p in self.active_policies] if self.active_policies else [],
            'policy_budget': self.policy_budget,
            'total_tax_collected': self.total_tax_collected
        })

        return obs

    def reset(self) -> Dict[str, float]:
        """Reset environment"""
        self.step_count = 0
        self.citizens = {}
        self.active_policies = []
        self.policy_budget = 0.0
        self.total_tax_collected = 0.0
        self.total_crimes = 0
        self.total_suffering = 0.0
        self.inequality_history = []
        self.trust_history = []

        self._initialize_citizens()

        return self._get_observation()

    def render(self) -> str:
        """Render environment as text"""
        lines = [f"\n=== Society Environment - Step {self.step_count} ==="]

        stats = self.get_statistics()

        lines.append(f"\nEconomic Indicators:")
        lines.append(f"  Avg Wealth: ${stats['avg_wealth']:.1f}")
        lines.append(f"  Inequality (Gini): {stats['inequality']:.3f}")
        lines.append(f"  Homeless Rate: {stats['homeless_rate']:.1%}")
        lines.append(f"  Hungry Rate: {stats['hungry_rate']:.1%}")

        lines.append(f"\nSocial Indicators:")
        lines.append(f"  Avg Happiness: {stats['avg_happiness']:.3f}")
        lines.append(f"  Avg Trust: {stats['avg_trust']:.3f}")
        lines.append(f"  Avg Suffering: {stats['avg_suffering']:.3f}")
        lines.append(f"  Crime Rate: {stats['crime_rate']:.4f}")
        lines.append(f"  Total Crimes: {stats['total_crimes']}")

        lines.append(f"\nPolicy:")
        if stats['active_policies']:
            lines.append(f"  Active: {', '.join(stats['active_policies'][-3:])}")
        else:
            lines.append(f"  Active: None")
        lines.append(f"  Budget: ${stats['policy_budget']:.1f}")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    env = SocietyEnv(num_citizens=100, seed=42)

    print("SocietyEnv Test:")
    print(env.render())

    # Run a few steps with policies
    for step in range(5):
        policy = np.random.choice([
            PolicyType.NO_POLICY,
            PolicyType.UNIVERSAL_BASIC_INCOME,
            PolicyType.WELFARE_SUPPORT
        ])

        obs, reward, done, info = env.step(policy)

    print(env.render())
    print(f"\nReward: {reward:.2f}")
    print("\n✓ SocietyEnv initialized successfully")
