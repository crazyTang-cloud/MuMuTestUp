"""
Git operations handler for Phase 0.

Handles repository reset to specific commit for consistent context.
"""
import logging
import subprocess
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class GitHandler:
    """
    Manages Git operations for the pipeline.
    
    Primary responsibility is to reset the repository to a specific commit
    to ensure consistent context for error analysis.
    """
    
    def __init__(self, repo_path: str):
        """
        Initialize Git handler.
        
        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = Path(repo_path).resolve()
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a Git repository: {self.repo_path}")
    
    def reset_to_commit(self, commit_id: str, force: bool = True) -> Tuple[bool, str]:
        """
        Reset repository to a specific commit.
        
        Args:
            commit_id: Git commit hash or reference
            force: If True, use --hard reset (default)
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        logger.info(f"Resetting repository to commit: {commit_id}")
        
        try:
            # First, check if commit exists
            check_cmd = ["git", "rev-parse", "--verify", commit_id]
            result = subprocess.run(
                check_cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                error_msg = f"Invalid commit ID '{commit_id}': {result.stderr.strip()}"
                logger.error(error_msg)
                return False, error_msg
            
            # Perform the reset
            reset_type = "--hard" if force else "--soft"
            reset_cmd = ["git", "reset", reset_type, commit_id]
            
            result = subprocess.run(
                reset_cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                success_msg = f"Successfully reset to commit {commit_id}"
                logger.info(success_msg)
                return True, success_msg
            else:
                error_msg = f"Git reset failed: {result.stderr.strip()}"
                logger.error(error_msg)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = "Git reset operation timed out"
            logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Git reset failed with exception: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_current_commit(self) -> Tuple[bool, str]:
        """
        Get current commit hash.
        
        Returns:
            Tuple of (success: bool, commit_hash: str)
        """
        try:
            cmd = ["git", "rev-parse", "HEAD"]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
                return True, commit_hash
            else:
                return False, result.stderr.strip()
                
        except Exception as e:
            return False, str(e)
    
    def check_clean_working_tree(self) -> Tuple[bool, str]:
        """
        Check if working tree is clean (no uncommitted changes).
        
        Returns:
            Tuple of (is_clean: bool, status: str)
        """
        try:
            cmd = ["git", "status", "--porcelain"]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                is_clean = len(output) == 0
                status = "clean" if is_clean else "dirty (uncommitted changes)"
                return is_clean, status
            else:
                return False, result.stderr.strip()
                
        except Exception as e:
            return False, str(e)

