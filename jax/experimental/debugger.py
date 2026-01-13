"""
JAX experimental debugger

Debugging utilities.
"""

def debug_print(msg, *args):
    """Debug print function."""
    print(f"DEBUG: {msg}", *args)

def breakpoint():
    """Debugging breakpoint."""
    import pdb; pdb.set_trace()

__all__ = ['debug_print', 'breakpoint']
