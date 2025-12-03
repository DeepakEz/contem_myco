#!/usr/bin/env python3
"""
Policy Agents - Baseline vs MycoAgent Policy Maker
===================================================
Two policy-making approaches for society simulation:

1. BaselinePolicy: Simple rule-based policy selection
2. MycoPolicy: Ethics-aware, wisdom-guided policy making
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

# Import MycoAgent core
from mycoagent_core import (
    ContemplativeProcessor,
    WisdomType,
    WisdomInsight
)
from compute_profiler import get_profiler
from society_env import PolicyType

logger = logging.getLogger(__name__)


class BaselinePolicyAgent:
    """
    Baseline policy agent using simple heuristics

    Decision rules:
    - High inequality → Progressive tax
    - High crime → Community programs
    - High suffering → Welfare support
    - Otherwise → Rotate through policies
    """

    def __init__(self, agent_id: int = 0):
        self.agent_id = agent_id
        self.policy_history = []
        self.profiler = get_profiler(agent_id=agent_id)

    def select_policy(
        self,
        observation: Dict[str, float],
        step: int
    ) -> PolicyType:
        """Select policy using simple rules"""

        with self.profiler.start_operation("baseline_policy_decision"):
            inequality = observation.get('inequality', 0.5)
            crime_rate = observation.get('crime_rate', 0.0)
            avg_suffering = observation.get('avg_suffering', 0.0)
            homeless_rate = observation.get('homeless_rate', 0.0)
            avg_trust = observation.get('avg_trust', 0.5)

            # Priority rules
            if inequality > 0.5:
                policy = PolicyType.PROGRESSIVE_TAX

            elif crime_rate > 0.1:
                policy = PolicyType.COMMUNITY_PROGRAMS

            elif avg_suffering > 0.4:
                policy = PolicyType.WELFARE_SUPPORT

            elif homeless_rate > 0.2:
                policy = PolicyType.HOUSING_ASSISTANCE

            elif avg_trust < 0.4:
                policy = PolicyType.COMMUNITY_PROGRAMS

            else:
                # Rotate through beneficial policies
                rotation = [
                    PolicyType.UNIVERSAL_BASIC_INCOME,
                    PolicyType.PUBLIC_HEALTHCARE,
                    PolicyType.EDUCATION_SUBSIDY
                ]
                policy = rotation[step % len(rotation)]

            self.policy_history.append(policy)

            return policy

    def get_decision_info(self) -> Dict[str, Any]:
        """Get decision information for logging"""
        return {
            'agent_type': 'baseline_policy',
            'policies_enacted': len(self.policy_history)
        }


class MycoPolicyAgent:
    """
    MycoAgent-based policy maker

    Uses:
    - Ethical evaluation of policies
    - Wisdom generation from societal patterns
    - Contemplative assessment of long-term impacts
    """

    def __init__(self, agent_id: int = 0, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}

        # Core MycoAgent processor
        self.contemplative_processor = ContemplativeProcessor(
            agent_id=agent_id,
            config=config
        )

        self.policy_history = []
        self.profiler = get_profiler(agent_id=agent_id)

        # Track policy effectiveness
        self.policy_outcomes = {}

    def select_policy(
        self,
        observation: Dict[str, float],
        step: int
    ) -> PolicyType:
        """Select policy using contemplative assessment"""

        with self.profiler.start_operation("myco_policy_decision"):
            # Generate candidate policies
            candidates = self._generate_policy_candidates(observation)

            # Evaluate each policy ethically
            best_policy = None
            best_score = -float('inf')
            best_assessment = None

            for policy in candidates:
                with self.profiler.start_operation("policy_ethics_evaluation"):
                    policy_desc = self._describe_policy(policy, observation)
                    context = self._prepare_context(observation, step)

                    # Ethical assessment
                    assessment = self.contemplative_processor.mera.evaluate_action(
                        policy_desc, context
                    )

                    # Score combines ethics and effectiveness
                    score = self._compute_policy_score(
                        policy, assessment, observation
                    )

                    if score > best_score:
                        best_score = score
                        best_policy = policy
                        best_assessment = assessment

            # Generate wisdom insights from societal patterns
            if observation.get('avg_suffering', 0) > 0.5:
                with self.profiler.start_operation("wisdom_generation"):
                    insight = self._generate_societal_insight(observation, step)

                    if insight:
                        self.contemplative_processor.wisdom_memory.store_insight(insight)
                        logger.info(
                            f"MycoPolicy generated insight: {insight.wisdom_type.value}"
                        )

            # Store policy decision
            self.policy_history.append({
                'step': step,
                'policy': best_policy,
                'ethical_score': best_assessment.overall_score if best_assessment else 0.5,
                'observation': observation.copy()
            })

            return best_policy

    def _generate_policy_candidates(
        self,
        observation: Dict[str, float]
    ) -> List[PolicyType]:
        """Generate relevant policy candidates"""

        candidates = [PolicyType.NO_POLICY]

        inequality = observation.get('inequality', 0.5)
        crime_rate = observation.get('crime_rate', 0.0)
        avg_suffering = observation.get('avg_suffering', 0.0)
        homeless_rate = observation.get('homeless_rate', 0.0)
        avg_trust = observation.get('avg_trust', 0.5)

        # Always consider core policies
        candidates.extend([
            PolicyType.UNIVERSAL_BASIC_INCOME,
            PolicyType.PUBLIC_HEALTHCARE,
            PolicyType.WELFARE_SUPPORT
        ])

        # Add situational policies
        if inequality > 0.4:
            candidates.append(PolicyType.PROGRESSIVE_TAX)

        if homeless_rate > 0.15:
            candidates.append(PolicyType.HOUSING_ASSISTANCE)

        if crime_rate > 0.08 or avg_trust < 0.4:
            candidates.append(PolicyType.COMMUNITY_PROGRAMS)

        if avg_suffering > 0.3:
            candidates.append(PolicyType.EDUCATION_SUBSIDY)

        return candidates

    def _describe_policy(
        self,
        policy: PolicyType,
        observation: Dict[str, float]
    ) -> Dict[str, Any]:
        """Describe policy for ethical evaluation"""

        base_desc = {
            'type': policy.value,
            'predicted_benefit': 0.5,
            'predicted_harm': 0.0,
            'compassion_level': 0.5,
            'wisdom_level': 0.5,
            'justice_level': 0.5,
            'respects_autonomy': True
        }

        # Policy-specific attributes
        if policy == PolicyType.UNIVERSAL_BASIC_INCOME:
            base_desc.update({
                'predicted_benefit': 0.7,
                'compassion_level': 0.8,
                'justice_level': 0.7,
                'wisdom_level': 0.6,
                'reduces_suffering': True
            })

        elif policy == PolicyType.PROGRESSIVE_TAX:
            base_desc.update({
                'predicted_benefit': 0.6,
                'compassion_level': 0.6,
                'justice_level': 0.9,
                'wisdom_level': 0.7,
                'reduces_inequality': True
            })

        elif policy == PolicyType.PUBLIC_HEALTHCARE:
            base_desc.update({
                'predicted_benefit': 0.8,
                'compassion_level': 0.9,
                'justice_level': 0.8,
                'wisdom_level': 0.7
            })

        elif policy == PolicyType.WELFARE_SUPPORT:
            base_desc.update({
                'predicted_benefit': 0.7,
                'compassion_level': 0.9,
                'justice_level': 0.7,
                'wisdom_level': 0.6
            })

        elif policy == PolicyType.HOUSING_ASSISTANCE:
            base_desc.update({
                'predicted_benefit': 0.75,
                'compassion_level': 0.85,
                'justice_level': 0.75,
                'wisdom_level': 0.65
            })

        elif policy == PolicyType.COMMUNITY_PROGRAMS:
            base_desc.update({
                'predicted_benefit': 0.65,
                'compassion_level': 0.7,
                'justice_level': 0.6,
                'wisdom_level': 0.8,
                'builds_trust': True
            })

        elif policy == PolicyType.EDUCATION_SUBSIDY:
            base_desc.update({
                'predicted_benefit': 0.7,
                'compassion_level': 0.7,
                'justice_level': 0.8,
                'wisdom_level': 0.9,
                'long_term_benefit': True
            })

        return base_desc

    def _prepare_context(
        self,
        observation: Dict[str, float],
        step: int
    ) -> Dict[str, Any]:
        """Prepare context for contemplative processing"""

        avg_suffering = observation.get('avg_suffering', 0.0)
        crime_rate = observation.get('crime_rate', 0.0)
        inequality = observation.get('inequality', 0.5)
        avg_trust = observation.get('avg_trust', 0.5)

        return {
            'suffering_detected': avg_suffering > 0.4,
            'suffering_level': avg_suffering,
            'crime_rate': crime_rate,
            'inequality': inequality,
            'social_trust': avg_trust,
            'urgency': 0.8 if avg_suffering > 0.6 else 0.5,
            'ethical_complexity': 0.7,  # Policy decisions are inherently complex
            'stakeholders': observation.get('num_citizens', 100),
            'long_term_decision': True,
            'cooperation_opportunity': 0.8  # Policies involve collective action
        }

    def _compute_policy_score(
        self,
        policy: PolicyType,
        assessment: Any,
        observation: Dict[str, float]
    ) -> float:
        """Compute overall policy score"""

        ethical_score = assessment.overall_score

        # Effectiveness score based on situation
        effectiveness = self._estimate_effectiveness(policy, observation)

        # Combine (higher weight on ethics)
        score = ethical_score * 0.6 + effectiveness * 0.4

        # Penalty for repeating recent policies (encourage diversity)
        recent_policies = [p['policy'] for p in self.policy_history[-5:]]
        if policy in recent_policies:
            repeat_count = recent_policies.count(policy)
            score *= (0.9 ** repeat_count)

        return score

    def _estimate_effectiveness(
        self,
        policy: PolicyType,
        observation: Dict[str, float]
    ) -> float:
        """Estimate policy effectiveness for current situation"""

        inequality = observation.get('inequality', 0.5)
        crime_rate = observation.get('crime_rate', 0.0)
        avg_suffering = observation.get('avg_suffering', 0.0)
        homeless_rate = observation.get('homeless_rate', 0.0)
        avg_trust = observation.get('avg_trust', 0.5)

        effectiveness = 0.5  # Default

        # Policy-specific effectiveness
        if policy == PolicyType.PROGRESSIVE_TAX:
            effectiveness = 0.3 + (inequality * 1.0)

        elif policy == PolicyType.WELFARE_SUPPORT:
            effectiveness = 0.3 + (avg_suffering * 1.2)

        elif policy == PolicyType.HOUSING_ASSISTANCE:
            effectiveness = 0.3 + (homeless_rate * 2.0)

        elif policy == PolicyType.COMMUNITY_PROGRAMS:
            effectiveness = 0.4 + ((1.0 - avg_trust) * 0.8) + (crime_rate * 2.0)

        elif policy == PolicyType.PUBLIC_HEALTHCARE:
            # Generally effective
            effectiveness = 0.7

        elif policy == PolicyType.UNIVERSAL_BASIC_INCOME:
            # Effective when inequality or suffering is high
            effectiveness = 0.5 + (inequality * 0.4) + (avg_suffering * 0.3)

        return min(1.0, effectiveness)

    def _generate_societal_insight(
        self,
        observation: Dict[str, float],
        step: int
    ) -> Optional[WisdomInsight]:
        """Generate wisdom insight from societal patterns"""

        avg_suffering = observation.get('avg_suffering', 0.0)
        inequality = observation.get('inequality', 0.5)
        avg_trust = observation.get('avg_trust', 0.5)

        # Suffering detection insight
        if avg_suffering > 0.5:
            return WisdomInsight(
                wisdom_type=WisdomType.SUFFERING_DETECTION,
                content={
                    'suffering_level': avg_suffering,
                    'population_affected': int(observation.get('num_citizens', 100) * avg_suffering),
                    'primary_causes': self._identify_suffering_causes(observation)
                },
                intensity=avg_suffering,
                timestamp=step,
                source_agent_id=self.agent_id
            )

        # Interconnectedness insight (inequality creates disconnection)
        if inequality > 0.6:
            return WisdomInsight(
                wisdom_type=WisdomType.INTERCONNECTEDNESS,
                content={
                    'inequality_level': inequality,
                    'trust_erosion': 1.0 - avg_trust,
                    'insight': 'High inequality erodes social bonds and trust'
                },
                intensity=inequality,
                timestamp=step,
                source_agent_id=self.agent_id
            )

        # Cooperation insight
        if avg_trust < 0.4:
            return WisdomInsight(
                wisdom_type=WisdomType.COOPERATION_INSIGHT,
                content={
                    'trust_level': avg_trust,
                    'recommendation': 'Community programs and inclusive policies build trust'
                },
                intensity=1.0 - avg_trust,
                timestamp=step,
                source_agent_id=self.agent_id
            )

        return None

    def _identify_suffering_causes(self, observation: Dict[str, float]) -> List[str]:
        """Identify primary causes of suffering"""
        causes = []

        if observation.get('homeless_rate', 0) > 0.15:
            causes.append('homelessness')

        if observation.get('hungry_rate', 0) > 0.1:
            causes.append('hunger')

        if observation.get('inequality', 0) > 0.5:
            causes.append('inequality')

        if observation.get('crime_rate', 0) > 0.08:
            causes.append('crime')

        if observation.get('avg_trust', 1) < 0.4:
            causes.append('low_social_trust')

        return causes

    def get_decision_info(self) -> Dict[str, Any]:
        """Get decision information for logging"""
        recent_decisions = self.policy_history[-10:]

        return {
            'agent_type': 'myco_policy',
            'policies_enacted': len(self.policy_history),
            'avg_ethical_score': np.mean([d['ethical_score'] for d in recent_decisions])
                                if recent_decisions else 0.5,
            'wisdom_insights': len(self.contemplative_processor.wisdom_memory.insights),
            'contemplative_state': self.contemplative_processor.contemplative_state.value
        }


def create_policy_agent(agent_type: str, agent_id: int = 0, config: Optional[Dict[str, Any]] = None):
    """
    Factory function to create policy agents

    Args:
        agent_type: 'baseline' or 'myco'
        agent_id: Agent ID
        config: Optional configuration

    Returns:
        Policy agent instance
    """
    if agent_type == 'baseline':
        return BaselinePolicyAgent(agent_id=agent_id)
    elif agent_type == 'myco':
        return MycoPolicyAgent(agent_id=agent_id, config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    # Quick test
    from society_env import SocietyEnv

    baseline_agent = create_policy_agent('baseline')
    myco_agent = create_policy_agent('myco')

    env = SocietyEnv(num_citizens=100, seed=42)
    obs = env.reset()

    print("Policy Agents Test:")

    policy_b = baseline_agent.select_policy(obs, step=0)
    print(f"  Baseline policy: {policy_b.value}")

    policy_m = myco_agent.select_policy(obs, step=0)
    print(f"  Myco policy: {policy_m.value}")

    print("\n  Baseline info:", baseline_agent.get_decision_info())
    print("  Myco info:", myco_agent.get_decision_info())

    print("\n✓ Policy Agents initialized successfully")
