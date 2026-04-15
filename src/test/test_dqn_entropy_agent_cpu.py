"""
Legacy compatibility wrapper for DQN+Entropy runs.

Device selection is no longer hard-coded in Python. Use the shell scripts to
choose CPU vs GPU, then invoke this module or the main entropy test module.
"""

from src.test.test_dqn_entropy_agent import main


if __name__ == "__main__":
    main()
