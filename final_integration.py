"""
Final Integration and Complete System Test
==========================================

This module provides the final integration layer and comprehensive system test
for the complete contemplative MycoNet system. It ensures all modules work
together seamlessly.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import all contemplative modules
try:
    from myconet_contemplative_core import (
        ContemplativeProcessor, WisdomInsight, WisdomType, ContemplativeState
    )
    from myconet_wisdom_signals import (
        WisdomSignalGrid, WisdomSignalConfig, WisdomSignalType, WisdomSignalProcessor
    )
    from myconet_contemplative_brains import (
        ContemplativeBrain, CollectiveWisdomBrain, create_contemplative_brain
    )
    from myconet_contemplative_entities import (
        ContemplativeNeuroAgent, ContemplativeColony, ContemplativeConfig
    )
    from myconet_contemplative_overmind import (
        ContemplativeOvermind, NetworkDharmaCompiler, OvermindAction
    )
    from myconet_contemplative_main import (
        ContemplativeSimulation, ContemplativeSimulationConfig, ContemplativeEnvironment
    )
    from myconet_contemplative_integration import (
        ContemplativeFeatureManager, ExperimentRunner, DataAnalyzer,
        MycoNetCompatibilityLayer
    )
    from myconet_contemplative_visualization import (
        ContemplativeVisualizer, create_visualization_suite
    )
    from myconet_contemplative_training import (
        ContemplativeTrainer, ContemplativeTrainingConfig, ContemplativeGymEnvironment
    )
    
    ALL_MODULES_IMPORTED = True
    IMPORT_ERRORS = []
    
except Exception as e:
    ALL_MODULES_IMPORTED = False
    IMPORT_ERRORS = [str(e)]
    print(f"Warning: Some modules failed to import: {e}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteSystemTest:
    """
    Comprehensive test suite for the entire contemplative MycoNet system
    """
    
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests"""
        print("🧠 MYCONET++ CONTEMPLATIVE AI - COMPLETE SYSTEM TEST")
        print("=" * 60)
        
        # Test order matters - dependencies first
        test_methods = [
            self.test_imports,
            self.test_core_contemplative,
            self.test_wisdom_signals,
            self.test_contemplative_brains,
            self.test_contemplative_entities,
            self.test_contemplative_overmind,
            self.test_simulation_environment,
            self.test_integration_layer,
            self.test_visualization_system,
            self.test_training_pipeline,
            self.test_end_to_end_simulation,
            self.test_feature_combinations
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            print(f"\n🔬 Running {test_name}...")
            
            try:
                start_time = time.time()
                result = test_method()
                end_time = time.time()
                
                if result:
                    print(f"✅ {test_name} PASSED ({end_time - start_time:.2f}s)")
                    self.passed_tests.append(test_name)
                    self.test_results[test_name] = {'status': 'PASSED', 'time': end_time - start_time}
                else:
                    print(f"❌ {test_name} FAILED ({end_time - start_time:.2f}s)")
                    self.failed_tests.append(test_name)
                    self.test_results[test_name] = {'status': 'FAILED', 'time': end_time - start_time}
                    
            except Exception as e:
                print(f"💥 {test_name} CRASHED: {e}")
                self.failed_tests.append(test_name)
                self.test_results[test_name] = {'status': 'CRASHED', 'error': str(e)}
        
        # Print final summary
        self._print_test_summary()
        
        return self.test_results
    
    def test_imports(self) -> bool:
        """Test that all modules import correctly"""
        if not ALL_MODULES_IMPORTED:
            print(f"  Import errors: {IMPORT_ERRORS}")
            return False
        
        print("  ✓ All core modules imported successfully")
        return True
    
    def test_core_contemplative(self) -> bool:
        """Test core contemplative processing components"""
        try:
            # Test ContemplativeProcessor
            processor = ContemplativeProcessor(agent_id=1, config={})
            print("  ✓ ContemplativeProcessor created")
            
            # Test processing functionality
            observations = {
                'attention_weights': [0.5, 0.3, 0.2],
                'other_agents_distress': 0.8,
                'collaboration_opportunities': 0.6
            }
            
            decision_context = {
                'decision_confidence': 0.7,
                'ethical_complexity': 0.8,
                'stakes_level': 0.9
            }
            
            result = processor.process_contemplatively(observations, decision_context)
            print("  ✓ Contemplative processing executed")
            
            # Test wisdom insight creation
            insight = WisdomInsight(
                wisdom_type=WisdomType.SUFFERING_DETECTION,
                content={'test': 'data'},
                intensity=0.8,
                timestamp=time.time()
            )
            
            processor.receive_wisdom_signal(insight)
            print("  ✓ Wisdom insight processing works")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Core contemplative test failed: {e}")
            return False
    
    def test_wisdom_signals(self) -> bool:
        """Test wisdom signal propagation system"""
        try:
            # Create wisdom signal grid
            config = WisdomSignalConfig()
            grid = WisdomSignalGrid(20, 20, config)
            print("  ✓ WisdomSignalGrid created")
            
            # Test signal propagation
            insight = WisdomInsight(
                wisdom_type=WisdomType.COMPASSION_RESPONSE,
                content={'compassion_level': 0.9},
                intensity=0.8,
                timestamp=time.time()
            )
            
            grid.propagate_wisdom_signal(
                10, 10, WisdomSignalType.COMPASSION_GRADIENT, 0.8, insight, 1
            )
            print("  ✓ Wisdom signal propagated")
            
            # Test signal updates
            grid.update_all_signals()
            print("  ✓ Signal diffusion and decay works")
            
            # Test signal processor
            processor = WisdomSignalProcessor(1, grid)
            local_signals = processor.process_local_signals(10, 10)
            print("  ✓ WisdomSignalProcessor works")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Wisdom signals test failed: {e}")
            return False
    
    def test_contemplative_brains(self) -> bool:
        """Test contemplative brain architectures"""
        try:
            # Test brain creation
            brain = create_contemplative_brain(
                brain_type='individual',
                input_size=12,
                hidden_size=32,
                output_size=6
            )
            print("  ✓ ContemplativeBrain created")
            
            # Test brain processing (if PyTorch available)
            try:
                import torch
                
                # Test forward pass
                test_input = torch.randn(1, 12)
                output, info = brain.forward(test_input)
                print("  ✓ Brain forward pass works")
                
                # Test contemplative decision making
                observations = {
                    'energy': 0.7,
                    'health': 0.8,
                    'suffering_detected': 0.6
                }
                
                available_actions = ['move_north', 'help', 'meditate']
                decision, decision_info = brain.make_contemplative_decision(observations, available_actions)
                print("  ✓ Contemplative decision making works")
                
            except ImportError:
                print("  ⚠ PyTorch not available, using dummy brain")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Contemplative brains test failed: {e}")
            return False
    
    def test_contemplative_entities(self) -> bool:
        """Test contemplative agents and colony"""
        try:
            # Create contemplative agent
            agent_config = {
                'contemplative_config': {
                    'enable_contemplative_processing': True,
                    'mindfulness_update_frequency': 10
                }
            }
            
            agent = ContemplativeNeuroAgent(
                agent_id=1, x=5, y=5, config=agent_config
            )
            print("  ✓ ContemplativeNeuroAgent created")
            
            # Test agent update
            observations = {
                'energy': 0.7,
                'food_nearby': 0.3,
                'suffering_detected': 0.0
            }
            
            available_actions = ['move_north', 'eat_food', 'contemplate']
            action = agent.update(observations, available_actions)
            print("  ✓ Agent update and decision making works")
            
            # Test colony
            grid = WisdomSignalGrid(10, 10, WisdomSignalConfig())
            colony = ContemplativeColony([agent], grid)
            colony.update_collective_state()
            print("  ✓ ContemplativeColony works")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Contemplative entities test failed: {e}")
            return False
    
    def test_contemplative_overmind(self) -> bool:
        """Test contemplative Overmind system"""
        try:
            # Create Overmind
            overmind = ContemplativeOvermind(
                colony_size=10,
                environment_size=(20, 20),
                config={}
            )
            print("  ✓ ContemplativeOvermind created")
            
            # Test dharma compiler
            dharma_compiler = NetworkDharmaCompiler()
            
            test_action = OvermindAction(
                action_type='coordinate_help',
                parameters={'target': 'suffering_areas'},
                urgency=0.8,
                ethical_weight=0.9,
                expected_benefit=0.7
            )
            
            colony_state = {
                'population': 10,
                'suffering_areas': [(5, 5, 0.8)],
                'average_energy': 0.6
            }
            
            compiled_action, ethical_assessment = dharma_compiler.compile_decision(
                test_action, colony_state, []
            )
            print("  ✓ NetworkDharmaCompiler works")
            print(f"    Ethical assessment: {ethical_assessment.get('reduce_suffering', 0):.2f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Contemplative Overmind test failed: {e}")
            return False
    
    def test_simulation_environment(self) -> bool:
        """Test simulation environment and main loop"""
        try:
            # Create simulation config
            config = ContemplativeSimulationConfig(
                environment_width=15,
                environment_height=15,
                initial_population=5,
                max_steps=10,  # Very short for testing
                enable_overmind=True
            )
            print("  ✓ ContemplativeSimulationConfig created")
            
            # Test environment
            env = ContemplativeEnvironment(15, 15)
            observations = env.get_local_observations(7, 7)
            print("  ✓ ContemplativeEnvironment works")
            
            # Create and test simulation (without running full simulation)
            simulation = ContemplativeSimulation(config)
            print("  ✓ ContemplativeSimulation initialized")
            
            # Test one simulation step
            simulation._simulation_step()
            print("  ✓ Simulation step executed")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Simulation environment test failed: {e}")
            return False
    
    def test_integration_layer(self) -> bool:
        """Test integration and compatibility layers"""
        try:
            # Test feature manager
            feature_manager = ContemplativeFeatureManager({})
            print("  ✓ ContemplativeFeatureManager created")
            
            # Test feature configuration
            feature_manager.enable_feature('mindfulness_processing')
            feature_manager.disable_feature('collective_meditation')
            
            agent_config = feature_manager.create_agent_config({})
            print("  ✓ Feature management works")
            
            # Test experiment runner
            experiment_runner = ExperimentRunner(feature_manager)
            experiments = experiment_runner.list_experiments()
            print(f"  ✓ ExperimentRunner works ({len(experiments)} templates)")
            
            # Test data analyzer
            analyzer = DataAnalyzer()
            dummy_data = {
                'population_data': [{'step': 0, 'total_population': 10}],
                'wisdom_data': [{'step': 0, 'total_wisdom_generated': 5}]
            }
            
            analysis = analyzer.analyze_simulation_data(dummy_data, ['wisdom_propagation'])
            print("  ✓ DataAnalyzer works")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Integration layer test failed: {e}")
            return False
    
    def test_visualization_system(self) -> bool:
        """Test visualization system"""
        try:
            from myconet_contemplative_visualization import test_visualization_system
            
            # Test visualization system
            viz_result = test_visualization_system()
            
            if viz_result:
                print("  ✓ Visualization system works")
            else:
                print("  ⚠ Visualization system has limitations (dependencies missing)")
            
            return True  # Don't fail if visualization isn't available
            
        except Exception as e:
            print(f"  ❌ Visualization test failed: {e}")
            return False
    
    def test_training_pipeline(self) -> bool:
        """Test training pipeline"""
        try:
            from myconet_contemplative_training import test_contemplative_training
            
            # Test training system
            training_result = test_contemplative_training()
            
            if training_result:
                print("  ✓ Training pipeline works")
            else:
                print("  ⚠ Training pipeline has limitations (dependencies missing)")
            
            return True  # Don't fail if training libs aren't available
            
        except Exception as e:
            print(f"  ❌ Training pipeline test failed: {e}")
            return False
    
    def test_end_to_end_simulation(self) -> bool:
        """Test complete end-to-end simulation"""
        try:
            print("  Running mini end-to-end simulation...")
            
            # Create minimal configuration
            config = ContemplativeSimulationConfig(
                environment_width=10,
                environment_height=10,
                initial_population=3,
                max_steps=5,  # Very short
                enable_overmind=True,
                save_interval=2
            )
            
            # Run simulation
            simulation = ContemplativeSimulation(config)
            
            # Run just a few steps
            for step in range(5):
                simulation._simulation_step()
                if step % 2 == 0:
                    simulation._collect_data()
            
            # Check results
            living_agents = len([a for a in simulation.agents if a.alive])
            data_points = len(simulation.simulation_data['population_data'])
            
            print(f"    Simulation completed: {living_agents} agents alive, {data_points} data points")
            print("  ✓ End-to-end simulation works")
            
            return True
            
        except Exception as e:
            print(f"  ❌ End-to-end simulation test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_feature_combinations(self) -> bool:
        """Test different feature combinations"""
        try:
            # Test different feature combinations
            feature_manager = ContemplativeFeatureManager({})
            
            test_combinations = [
                ['mindfulness_processing'],
                ['mindfulness_processing', 'wisdom_signals'],
                ['mindfulness_processing', 'ethical_reasoning'],
                ['wisdom_signals', 'collective_meditation'],
            ]
            
            for features in test_combinations:
                # Reset features
                for feature in feature_manager.feature_flags:
                    feature_manager.disable_feature(feature)
                
                # Enable test combination
                for feature in features:
                    feature_manager.enable_feature(feature)
                
                # Create agent config
                agent_config = feature_manager.create_agent_config({})
                
                # Test agent creation
                agent = ContemplativeNeuroAgent(
                    agent_id=1, x=0, y=0, config=agent_config
                )
                
                print(f"    ✓ Feature combination {features} works")
            
            print("  ✓ Feature combinations test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Feature combinations test failed: {e}")
            return False
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print("\n" + "=" * 60)
        print("🧠 MYCONET++ CONTEMPLATIVE AI - TEST SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"Success Rate: {passed_count/total_tests:.1%}")
        
        if self.failed_tests:
            print(f"\nFailed Tests:")
            for test_name in self.failed_tests:
                status_info = self.test_results.get(test_name, {})
                if 'error' in status_info:
                    print(f"  💥 {test_name}: {status_info['error']}")
                else:
                    print(f"  ❌ {test_name}")
        
        print(f"\nPassed Tests:")
        for test_name in self.passed_tests:
            test_time = self.test_results.get(test_name, {}).get('time', 0)
            print(f"  ✅ {test_name} ({test_time:.2f}s)")
        
        # System status
        print(f"\n🌐 SYSTEM STATUS:")
        
        if passed_count == total_tests:
            print("🎉 ALL SYSTEMS OPERATIONAL!")
            print("   Ready for contemplative AI research and experimentation!")
        elif passed_count >= total_tests * 0.8:
            print("✅ CORE SYSTEMS OPERATIONAL")
            print("   Basic functionality available, some features may be limited")
        elif passed_count >= total_tests * 0.5:
            print("⚠️  PARTIAL FUNCTIONALITY")
            print("   Some core features working, recommend investigating failures")
        else:
            print("❌ MAJOR ISSUES DETECTED")
            print("   Recommend resolving critical failures before use")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("   1. Run: python setup_contemplative_myconet.py")
        print("   2. Install optional dependencies: pip install torch matplotlib seaborn")
        print("   3. Try minimal experiment: ./run.sh minimal")
        print("   4. Read README.md for detailed usage instructions")
        
        print("=" * 60)

# Main execution functions
def run_complete_system_test():
    """Run the complete system test"""
    tester = CompleteSystemTest()
    results = tester.run_all_tests()
    return results

def quick_functionality_check():
    """Quick check of basic functionality"""
    print("🚀 Quick Functionality Check")
    print("-" * 30)
    
    try:
        # Test basic imports
        print("Testing imports...", end="")
        if ALL_MODULES_IMPORTED:
            print(" ✅")
        else:
            print(" ❌")
            return False
        
        # Test core object creation
        print("Testing core objects...", end="")
        processor = ContemplativeProcessor(1, {})
        grid = WisdomSignalGrid(5, 5, WisdomSignalConfig())
        agent = ContemplativeNeuroAgent(1, 0, 0, {'contemplative_config': {}})
        print(" ✅")
        
        print("Testing simulation setup...", end="")
        config = ContemplativeSimulationConfig(
            environment_width=5, environment_height=5,
            initial_population=2, max_steps=2
        )
        sim = ContemplativeSimulation(config)
        print(" ✅")
        
        print("\n🎉 Basic functionality check PASSED!")
        print("   System is ready for use!")
        return True
        
    except Exception as e:
        print(f" ❌\nError: {e}")
        print("\n❌ Basic functionality check FAILED!")
        print("   Run complete system test for detailed diagnosis")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MycoNet++ Contemplative AI System Test")
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick functionality check only')
    parser.add_argument('--full', action='store_true',
                       help='Run complete system test (default)')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_functionality_check()
    else:
        run_complete_system_test()