#!/usr/bin/env python3
"""Test script to identify which agent tab build function fails."""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tkinter as tk
    from tkinter import ttk
    print("✓ tkinter imported successfully")
except ImportError as e:
    print(f"✗ tkinter import failed: {e}")
    sys.exit(1)

# Import GUI components
try:
    from src.gui import DeeconGui
    print("✓ DeeconGui imported successfully")
except Exception as e:
    print(f"✗ DeeconGui import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try to run GUI initialization
try:
    root = tk.Tk()
    print("✓ Tk window created")
    
    gui = DeeconGui(root)
    print("✓ DeeconGui initialized successfully")
    
    # Check if tabs are present
    if hasattr(gui, 'tabs') and gui.tabs:
        tab_count = len(gui.tabs.tabs())
        print(f"✓ Tabs found: {tab_count}")
        for i, tab_id in enumerate(gui.tabs.tabs()):
            tab_text = gui.tabs.tab(tab_id, "text")
            print(f"  Tab {i}: '{tab_text}'")
    else:
        print("✗ No tabs found")
    
    # Check agent notebook
    if hasattr(gui, 'agents_notebook') and gui.agents_notebook:
        agent_tab_count = len(gui.agents_notebook.tabs())
        print(f"✓ Agent notebook created with {agent_tab_count} tabs")
    else:
        print("✗ Agent notebook not created or not accessible")
        
except Exception as e:
    print(f"✗ GUI initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
root.destroy()
