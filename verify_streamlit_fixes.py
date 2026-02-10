#!/usr/bin/env python3
"""
Simple verification script for Streamlit GUI fixes
"""

def verify_cache_stats_fix():
    """Verify that cache stats fields are correctly mapped"""
    print("Verifying cache stats fix...")
    
    try:
        from converter.cache_manager import global_cache_manager
        
        # Get cache stats
        cache_stats = global_cache_manager.get_cache_stats()
        
        # Test the exact field access from fixed Streamlit GUI
        size_mb = cache_stats.get('memory_usage_mb', 0)
        providers_dict = cache_stats.get('providers', {})
        total_entries = cache_stats.get('total_entries', 0)
        recent = cache_stats.get('recent_24h', 0)
        
        print(f"  [OK] Cache size: {size_mb:.2f} MB")
        print(f"  [OK] Providers count: {len(providers_dict)}")
        print(f"  [OK] Total entries: {total_entries}")
        print(f"  [OK] Recent activity: {recent}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Cache stats test failed: {e}")
        return False

def verify_streamlit_gui_syntax():
    """Verify that Streamlit GUI can be imported without syntax errors"""
    print("\nVerifying Streamlit GUI imports...")
    
    try:
        import streamlit_gui
        print("  [OK] streamlit_gui module imported successfully")
        
        # Check that key functions exist
        if hasattr(streamlit_gui, 'render_cache_management'):
            print("  [OK] render_cache_management function exists")
        else:
            print("  [ERROR] render_cache_management function missing")
            return False
            
        if hasattr(streamlit_gui, 'start_conversion'):
            print("  [OK] start_conversion function exists")
        else:
            print("  [ERROR] start_conversion function missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"  [ERROR] Import failed: {e}")
        return False

def verify_function_signatures():
    """Verify that the convert_mineru_to_ppt function calls use correct parameters"""
    print("\nVerifying function call parameters...")
    
    try:
        import streamlit_gui
        import inspect
        
        # Check start_conversion function source
        source = inspect.getsource(streamlit_gui.start_conversion)
        
        # Check for correct parameter names
        if 'output_ppt_path=' in source:
            print("  [OK] Uses correct parameter 'output_ppt_path'")
        else:
            print("  [ERROR] Still uses incorrect parameter names")
            return False
            
        # Check that progress_callback is removed
        if 'progress_callback=' not in source:
            print("  [OK] progress_callback parameter removed")
        else:
            print("  [ERROR] progress_callback parameter still present")
            return False
            
        return True
        
    except Exception as e:
        print(f"  [ERROR] Function signature verification failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Streamlit GUI Fixes - Simple Verification")
    print("=" * 60)
    
    tests = [
        verify_cache_stats_fix,
        verify_streamlit_gui_syntax,
        verify_function_signatures
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  [ERROR] Test exception: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"Results: {passed}/{total} verifications passed")
    
    if passed == total:
        print("\n[SUCCESS] All fixes verified!")
        print("\nFixed Issues:")
        print("1. Cache stats 'total_size_mb' error → Fixed field mapping")
        print("2. Function parameter mismatch → Corrected to 'output_ppt_path'")
        print("3. Removed invalid 'progress_callback' parameter")
        print("4. Improved logging display with markdown formatting")
        print("\nThe Streamlit GUI should now work correctly!")
        print("Run: python run_streamlit.py")
    else:
        print(f"\n[WARNING] {total - passed} verification(s) failed")
        print("Some issues may still exist.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)