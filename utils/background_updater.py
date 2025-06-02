#!/usr/bin/env python3
"""
Background Update Manager for Space Debris Dashboard
Handles non-blocking data updates and smart notifications
"""

import threading
import time
from datetime import datetime, timedelta
import streamlit as st
from utils.database import (
    is_data_fresh, 
    populate_real_data_force_refresh, 
    get_cached_objects_count,
    get_metadata_value,
    set_metadata_value
)

class BackgroundUpdateManager:
    """Manages background data updates and notifications."""
    
    def __init__(self):
        self.update_thread = None
        self.is_running = False
        self.last_check_time = None
        self.update_interval_hours = 2
        self.notification_queue = []
        
    def start_background_updates(self):
        """Start the background update system."""
        if self.is_running:
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(target=self._background_update_loop, daemon=True)
        self.update_thread.start()
        print("🔄 Background update system started")
        
    def stop_background_updates(self):
        """Stop the background update system."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
        print("⏹️ Background update system stopped")
        
    def _background_update_loop(self):
        """Main background update loop."""
        while self.is_running:
            try:
                self.last_check_time = datetime.now()
                
                # Check if data needs updating
                if not is_data_fresh(max_age_hours=self.update_interval_hours):
                    print("🔄 Background: Data is stale, attempting refresh...")
                    self._attempt_background_refresh()
                else:
                    print("✅ Background: Data is fresh, no update needed")
                
                # Sleep for 30 minutes before next check
                time.sleep(30 * 60)  # 30 minutes
                
            except Exception as e:
                print(f"⚠️ Background update error: {e}")
                time.sleep(60)  # Wait 1 minute on error
                
    def _attempt_background_refresh(self):
        """Attempt to refresh data in background."""
        try:
            # Notify start of update
            self.add_notification("🔄 Background data update started...", "info")
            
            # Perform the update
            success = populate_real_data_force_refresh()
            
            if success:
                # Update metadata
                set_metadata_value('last_background_update', datetime.now().isoformat())
                
                # Notify success
                self.add_notification("✅ Fresh space debris data available! Refresh to see updates.", "success")
                print("✅ Background update completed successfully")
            else:
                self.add_notification("⚠️ Background update failed, using cached data", "warning")
                
        except Exception as e:
            print(f"❌ Background refresh failed: {e}")
            self.add_notification("❌ Background update failed, check connectivity", "error")
            
    def add_notification(self, message, type="info"):
        """Add a notification to the queue."""
        notification = {
            'message': message,
            'type': type,
            'timestamp': datetime.now(),
            'id': f"notif_{len(self.notification_queue)}"
        }
        self.notification_queue.append(notification)
        
        # Keep only last 5 notifications
        if len(self.notification_queue) > 5:
            self.notification_queue = self.notification_queue[-5:]
            
    def get_notifications(self, max_age_minutes=10):
        """Get recent notifications."""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        return [n for n in self.notification_queue if n['timestamp'] > cutoff_time]
        
    def clear_notifications(self):
        """Clear all notifications."""
        self.notification_queue = []
        
    def get_status(self):
        """Get current background update status."""
        return {
            'is_running': self.is_running,
            'last_check': self.last_check_time,
            'interval_hours': self.update_interval_hours,
            'cached_objects': get_cached_objects_count(),
            'data_fresh': is_data_fresh(max_age_hours=self.update_interval_hours),
            'last_update': get_metadata_value('last_celestrak_download', 'Never'),
            'last_background_update': get_metadata_value('last_background_update', 'Never')
        }

# Global instance
_background_manager = None

def get_background_manager():
    """Get the global background update manager."""
    global _background_manager
    if _background_manager is None:
        _background_manager = BackgroundUpdateManager()
    return _background_manager

def start_background_updates():
    """Start background updates."""
    manager = get_background_manager()
    manager.start_background_updates()
    return manager

def stop_background_updates():
    """Stop background updates."""
    manager = get_background_manager()
    manager.stop_background_updates()

def get_update_status():
    """Get current update status."""
    manager = get_background_manager()
    return manager.get_status()

def get_recent_notifications():
    """Get recent notifications."""
    manager = get_background_manager()
    return manager.get_notifications()

def clear_all_notifications():
    """Clear all notifications."""
    manager = get_background_manager()
    manager.clear_notifications() 