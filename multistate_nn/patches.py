"""
Patch functions to fix issues with external dependencies.

These patches are automatically applied when the module is imported.
"""

import logging

logger = logging.getLogger(__name__)

def patch_torchdiffeq():
    """
    Fix the rtol/atol duplication issue in torchdiffeq.odeint.
    
    The issue occurs because torchdiffeq passes both explicit rtol/atol parameters
    AND any rtol/atol in the options dictionary to the ODE solver, which causes a
    "got multiple values for keyword argument" error.
    """
    try:
        from torchdiffeq import odeint as original_odeint
        
        # Create a safe wrapper function
        def safe_odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
            """Wrapper around odeint that prevents duplicate rtol/atol."""
            # Create a clean copy of options without rtol/atol
            if options is not None:
                clean_options = {k: v for k, v in options.items() if k not in ['rtol', 'atol']}
            else:
                clean_options = None
            
            # Call original odeint with clean options
            return original_odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=clean_options, event_fn=event_fn)
        
        # Apply the patch to our modules that use odeint
        import multistate_nn.models
        multistate_nn.models.odeint = safe_odeint
        
        # Also patch the Bayesian module if it exists
        try:
            import multistate_nn.extensions.bayesian
            multistate_nn.extensions.bayesian.odeint = safe_odeint
            logger.info("Successfully patched torchdiffeq.odeint in both models and extensions")
        except ImportError:
            logger.info("Successfully patched torchdiffeq.odeint in models module only")
            
        return True
    except Exception as e:
        logger.warning(f"Failed to patch torchdiffeq: {str(e)}")
        return False

def apply_all_patches():
    """Apply all available patches."""
    patches = {
        "torchdiffeq": patch_torchdiffeq
    }
    
    results = {}
    for name, patch_func in patches.items():
        results[name] = patch_func()
    
    return results