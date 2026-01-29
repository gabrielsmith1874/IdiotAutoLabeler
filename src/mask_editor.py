"""
Mask Touch-Up Editor (Tkinter Version)
======================================
Quick tool to cycle through masked images and edit masks.

Controls:
- Left/Right Arrow or A/D: Navigate between images
- Left Mouse + Drag: Draw on mask (add magenta)
- Right Mouse + Drag: Erase mask (restore original)
- Slider or Mouse Wheel: Adjust brush size
- Q: Toggle mask visibility (show/hide)
- S: Save current image
- R: Reset current mask (reload from file)
- Escape: Quit

Author: Auto-Labeler Project
"""

import os
import glob
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    "data_dir": "data/auto_labeled",
    "window_width": 1400,
    "window_height": 850,
    "brush_size": 15,
    "min_brush": 3,
    "max_brush": 80,
}

MAGENTA = (255, 0, 255)

class MaskEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Touch-Up Editor")
        self.root.geometry(f"{CONFIG['window_width']}x{CONFIG['window_height']}")
        self.root.configure(bg='#1e1e1e')
        
        # Find all masked images
        pattern = os.path.join(CONFIG["data_dir"], "*_masked.png")
        self.masked_paths = sorted(glob.glob(pattern))
        
        if not self.masked_paths:
            messagebox.showerror("Error", f"No masked images found in {CONFIG['data_dir']}")
            root.destroy()
            return
        
        self.current_idx = 0
        self.brush_size = tk.IntVar(value=CONFIG["brush_size"])
        self.modified = False
        self.drawing = False
        self.erasing = False
        self.last_pos = None
        self.mask_visible = True
        self.cursor_id = None  # For brush cursor
        self.pending_update = False  # Throttle updates
        
        # Setup UI
        self.setup_ui()
        self.bind_events()
        self.load_current_image()
    
    def setup_ui(self):
        # Top toolbar
        self.toolbar = tk.Frame(self.root, bg='#2d2d2d', height=50)
        self.toolbar.pack(fill=tk.X, side=tk.TOP)
        
        # Navigation buttons
        self.prev_btn = tk.Button(
            self.toolbar, text="◀ Prev", command=lambda: self.navigate(-1),
            bg='#444', fg='white', font=('Arial', 10), padx=15
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5, pady=8)
        
        self.next_btn = tk.Button(
            self.toolbar, text="Next ▶", command=lambda: self.navigate(1),
            bg='#444', fg='white', font=('Arial', 10), padx=15
        )
        self.next_btn.pack(side=tk.LEFT, padx=5, pady=8)
        
        # Separator
        tk.Label(self.toolbar, text="  |  ", fg='#666', bg='#2d2d2d').pack(side=tk.LEFT)
        
        # Brush size slider
        tk.Label(self.toolbar, text="Brush:", fg='white', bg='#2d2d2d', font=('Arial', 10)).pack(side=tk.LEFT, padx=(10, 5))
        
        self.brush_slider = ttk.Scale(
            self.toolbar, from_=CONFIG["min_brush"], to=CONFIG["max_brush"],
            orient=tk.HORIZONTAL, length=150, variable=self.brush_size
        )
        self.brush_slider.pack(side=tk.LEFT, padx=5)
        
        self.brush_label = tk.Label(
            self.toolbar, textvariable=self.brush_size, fg='#00ff00', bg='#2d2d2d', 
            font=('Consolas', 11, 'bold'), width=3
        )
        self.brush_label.pack(side=tk.LEFT, padx=5)
        
        # Separator
        tk.Label(self.toolbar, text="  |  ", fg='#666', bg='#2d2d2d').pack(side=tk.LEFT)
        
        # Toggle mask button
        self.mask_btn = tk.Button(
            self.toolbar, text="Toggle Mask (Q)", command=self.toggle_mask,
            bg='#ff00ff', fg='white', font=('Arial', 10), padx=10
        )
        self.mask_btn.pack(side=tk.LEFT, padx=5, pady=8)
        
        # Save button
        self.save_btn = tk.Button(
            self.toolbar, text="💾 Save (S)", command=self.save_current,
            bg='#2a7d2a', fg='white', font=('Arial', 10, 'bold'), padx=15
        )
        self.save_btn.pack(side=tk.LEFT, padx=10, pady=8)
        
        # Reset button
        self.reset_btn = tk.Button(
            self.toolbar, text="↺ Reset (R)", command=self.load_current_image,
            bg='#7d2a2a', fg='white', font=('Arial', 10), padx=10
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5, pady=8)
        
        # Clear All button
        self.clear_btn = tk.Button(
            self.toolbar, text="✕ Clear All (C)", command=self.clear_all_mask,
            bg='#7d5a2a', fg='white', font=('Arial', 10), padx=10
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=8)
        
        # Image counter on right
        self.counter_label = tk.Label(
            self.toolbar, text="", fg='white', bg='#2d2d2d', font=('Consolas', 11)
        )
        self.counter_label.pack(side=tk.RIGHT, padx=15)
        
        # Main frame for canvas
        self.main_frame = tk.Frame(self.root, bg='#1e1e1e')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image
        self.canvas = tk.Canvas(self.main_frame, bg='#2d2d2d', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_frame = tk.Frame(self.root, bg='#333333', height=30)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            self.status_frame, text="", fg='white', bg='#333333', 
            font=('Consolas', 10), anchor='w'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.help_label = tk.Label(
            self.status_frame, 
            text="LMB: Draw | RMB: Erase | Q: Toggle Mask | S: Save | R: Reset | Esc: Quit",
            fg='#888888', bg='#333333', font=('Consolas', 9), anchor='e'
        )
        self.help_label.pack(side=tk.RIGHT, padx=10)
    
    def bind_events(self):
        # Key bindings
        self.root.bind('<Left>', lambda e: self.navigate(-1))
        self.root.bind('<Right>', lambda e: self.navigate(1))
        self.root.bind('<a>', lambda e: self.navigate(-1))
        self.root.bind('<d>', lambda e: self.navigate(1))
        self.root.bind('<s>', lambda e: self.save_current())
        self.root.bind('<Control-s>', lambda e: self.save_current())
        self.root.bind('<r>', lambda e: self.load_current_image())
        self.root.bind('<q>', lambda e: self.toggle_mask())
        self.root.bind('<c>', lambda e: self.clear_all_mask())
        self.root.bind('<Escape>', lambda e: self.quit_app())
        self.root.bind('<plus>', lambda e: self.adjust_brush(5))
        self.root.bind('<minus>', lambda e: self.adjust_brush(-5))
        self.root.bind('<equal>', lambda e: self.adjust_brush(5))
        
        # Mouse bindings
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<Button-3>', self.start_erase)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<B3-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        self.canvas.bind('<ButtonRelease-3>', self.stop_draw)
        self.canvas.bind('<MouseWheel>', self.on_scroll)
        self.canvas.bind('<Motion>', self.update_cursor)
        self.canvas.bind('<Leave>', self.hide_cursor)
        
        # Window resize
        self.canvas.bind('<Configure>', self.on_resize)
        
        # Close window handler
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
    
    def load_current_image(self):
        """Load the current masked and unmasked images."""
        masked_path = self.masked_paths[self.current_idx]
        unmasked_path = masked_path.replace("_masked.png", "_unmasked.png")
        
        # Load images as numpy arrays for fast operations
        self.masked_pil = Image.open(masked_path).convert("RGB")
        self.unmasked_pil = Image.open(unmasked_path).convert("RGB")
        
        self.orig_size = self.masked_pil.size  # (W, H)
        
        # Create working arrays (numpy for speed)
        self.working_arr = np.array(self.masked_pil)
        self.unmasked_arr = np.array(self.unmasked_pil)
        
        self.modified = False
        self.update_display()
        self.update_status()
    
    def toggle_mask(self):
        """Toggle mask visibility on/off."""
        self.mask_visible = not self.mask_visible
        if self.mask_visible:
            self.mask_btn.config(bg='#ff00ff', text="Toggle Mask (Q)")
        else:
            self.mask_btn.config(bg='#666666', text="Mask Hidden (Q)")
        self.update_display()
    
    def clear_all_mask(self):
        """Clear all mask pixels (restore to unmasked image)."""
        self.working_arr = self.unmasked_arr.copy()
        self.modified = True
        self.update_display()
        self.update_status()
    
    def update_display(self):
        """Update the canvas display."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return
        
        img_w, img_h = self.orig_size
        
        # Scale to fit
        scale = min(canvas_w / img_w, canvas_h / img_h)
        self.scale = scale
        self.display_size = (int(img_w * scale), int(img_h * scale))
        
        # Center offset
        self.offset_x = (canvas_w - self.display_size[0]) // 2
        self.offset_y = (canvas_h - self.display_size[1]) // 2
        
        # Choose which array to display
        if self.mask_visible:
            display_arr = self.working_arr
        else:
            display_arr = self.unmasked_arr
        
        # Create display image
        display_pil = Image.fromarray(display_arr)
        display_img = display_pil.resize(self.display_size, Image.NEAREST)  # NEAREST is faster
        self.photo = ImageTk.PhotoImage(display_img)
        
        self.canvas.delete("image")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor='nw', image=self.photo, tags="image")
        
        self.pending_update = False
    
    def update_cursor(self, event):
        """Draw brush cursor at mouse position."""
        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
        
        r = self.brush_size.get()
        self.cursor_id = self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            outline='#00ff00', width=2, tags="cursor"
        )
    
    def hide_cursor(self, event):
        """Hide brush cursor when mouse leaves canvas."""
        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
            self.cursor_id = None
    
    def screen_to_image(self, x, y):
        """Convert screen coordinates to image coordinates."""
        ix = int((x - self.offset_x) / self.scale)
        iy = int((y - self.offset_y) / self.scale)
        
        # Clamp to image bounds
        ix = max(0, min(self.orig_size[0] - 1, ix))
        iy = max(0, min(self.orig_size[1] - 1, iy))
        
        return ix, iy
    
    def draw_circle_fast(self, cx, cy, erase=False):
        """Draw or erase a circle using numpy (fast)."""
        r = int(self.brush_size.get() / self.scale)
        if r < 1:
            r = 1
        
        h, w = self.working_arr.shape[:2]
        
        # Compute bounding box
        y1 = max(0, cy - r)
        y2 = min(h, cy + r + 1)
        x1 = max(0, cx - r)
        x2 = min(w, cx + r + 1)
        
        # Create local coordinate grids
        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
        
        if erase:
            # Restore from unmasked
            self.working_arr[y1:y2, x1:x2][mask] = self.unmasked_arr[y1:y2, x1:x2][mask]
        else:
            # Apply magenta
            self.working_arr[y1:y2, x1:x2][mask] = MAGENTA
        
        self.modified = True
    
    def start_draw(self, event):
        self.drawing = True
        self.erasing = False
        ix, iy = self.screen_to_image(event.x, event.y)
        self.draw_circle_fast(ix, iy, erase=False)
        self.last_pos = (ix, iy)
        self.schedule_update()
        self.update_status()
    
    def start_erase(self, event):
        self.erasing = True
        self.drawing = False
        ix, iy = self.screen_to_image(event.x, event.y)
        self.draw_circle_fast(ix, iy, erase=True)
        self.last_pos = (ix, iy)
        self.schedule_update()
        self.update_status()
    
    def on_drag(self, event):
        """Handle mouse drag for drawing/erasing."""
        if not (self.drawing or self.erasing):
            return
        
        ix, iy = self.screen_to_image(event.x, event.y)
        
        # Draw line from last pos to current
        if self.last_pos:
            self.draw_line_fast(self.last_pos[0], self.last_pos[1], ix, iy, erase=self.erasing)
        
        self.last_pos = (ix, iy)
        self.update_cursor(event)
        self.schedule_update()
    
    def draw_line_fast(self, x1, y1, x2, y2, erase=False):
        """Draw a line of circles between two points."""
        dist = max(abs(x2 - x1), abs(y2 - y1))
        if dist == 0:
            self.draw_circle_fast(x1, y1, erase)
            return
        
        # Sample fewer points for speed
        step = max(1, dist // 3)
        for i in range(0, dist + 1, step):
            t = i / dist
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            self.draw_circle_fast(x, y, erase)
    
    def schedule_update(self):
        """Throttle display updates for performance."""
        if not self.pending_update:
            self.pending_update = True
            self.root.after(30, self.update_display)  # ~33 FPS
    
    def stop_draw(self, event):
        self.drawing = False
        self.erasing = False
        self.last_pos = None
        self.update_display()  # Final full update
    
    def on_scroll(self, event):
        if event.delta > 0:
            self.adjust_brush(5)
        else:
            self.adjust_brush(-5)
        self.update_cursor(event)
    
    def adjust_brush(self, delta):
        new_val = max(CONFIG["min_brush"], min(CONFIG["max_brush"], self.brush_size.get() + delta))
        self.brush_size.set(new_val)
    
    def on_resize(self, event):
        self.root.after(100, self.update_display)
    
    def update_status(self):
        mod_text = " *UNSAVED*" if self.modified else ""
        filename = os.path.basename(self.masked_paths[self.current_idx])
        self.status_label.config(text=f"{filename}{mod_text}")
        self.counter_label.config(text=f"[{self.current_idx + 1}/{len(self.masked_paths)}]")
        
        if self.modified:
            self.save_btn.config(bg='#ff6600')
        else:
            self.save_btn.config(bg='#2a7d2a')
    
    def save_current(self):
        """Save the current masked image."""
        masked_path = self.masked_paths[self.current_idx]
        Image.fromarray(self.working_arr).save(masked_path)
        self.modified = False
        self.update_status()
        print(f"Saved: {masked_path}")
    
    def navigate(self, delta):
        """Navigate to next/previous image."""
        if self.modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes", 
                "You have unsaved changes. Save before navigating?"
            )
            if result is None:
                return
            elif result:
                self.save_current()
        
        self.current_idx = (self.current_idx + delta) % len(self.masked_paths)
        self.load_current_image()
    
    def quit_app(self):
        if self.modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes", 
                "You have unsaved changes. Save before quitting?"
            )
            if result is None:
                return
            elif result:
                self.save_current()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MaskEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
