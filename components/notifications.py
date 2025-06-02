#!/usr/bin/env python3
"""
Notification Component for Space Debris Dashboard
Displays background update notifications and system status
"""

import streamlit as st
from datetime import datetime, timedelta
from utils.background_updater import get_recent_notifications, clear_all_notifications, get_update_status

def show_notifications():
    """Display recent notifications in the sidebar or main area."""
    notifications = get_recent_notifications()
    
    if not notifications:
        return False  # No notifications to show
    
    # Create notification area
    notification_container = st.container()
    
    with notification_container:
        # Notification header
        notif_header_col, clear_col = st.columns([3, 1])
        
        with notif_header_col:
            st.markdown("### 🔔 System Notifications")
        
        with clear_col:
            if st.button("🗑️ Clear", help="Clear all notifications", key="clear_notifications"):
                clear_all_notifications()
                st.rerun()
        
        # Display notifications
        for notif in reversed(notifications):  # Show newest first
            time_ago = _get_time_ago(notif['timestamp'])
            
            # Choose notification style based on type
            if notif['type'] == 'success':
                st.success(f"✅ {notif['message']} • {time_ago}")
            elif notif['type'] == 'warning':
                st.warning(f"⚠️ {notif['message']} • {time_ago}")
            elif notif['type'] == 'error':
                st.error(f"❌ {notif['message']} • {time_ago}")
            else:  # info
                st.info(f"ℹ️ {notif['message']} • {time_ago}")
    
    return True  # Notifications were shown

def show_background_status():
    """Show background update status in sidebar."""
    status = get_update_status()
    
    # Background status section
    st.markdown("### 🔄 Background Updates")
    
    # Status indicators
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        running_status = "🟢 Active" if status['is_running'] else "🔴 Inactive"
        st.metric("Status", running_status)
    
    with status_col2:
        fresh_status = "✅ Fresh" if status['data_fresh'] else "🔄 Updating"
        st.metric("Data", fresh_status)
    
    # Detailed status
    with st.expander("📊 Update Details", expanded=False):
        st.write(f"**Cached Objects:** {status['cached_objects']:,}")
        st.write(f"**Check Interval:** {status['interval_hours']} hours")
        
        # Last check time
        if status['last_check']:
            last_check_ago = _get_time_ago(status['last_check'])
            st.write(f"**Last Check:** {last_check_ago}")
        else:
            st.write("**Last Check:** Not started")
        
        # Last update time
        if status['last_update'] != 'Never':
            try:
                last_update_time = datetime.fromisoformat(status['last_update'])
                last_update_ago = _get_time_ago(last_update_time)
                st.write(f"**Last Data Update:** {last_update_ago}")
            except:
                st.write(f"**Last Data Update:** {status['last_update']}")
        else:
            st.write("**Last Data Update:** Never")
        
        # Background update time
        if status['last_background_update'] != 'Never':
            try:
                bg_update_time = datetime.fromisoformat(status['last_background_update'])
                bg_update_ago = _get_time_ago(bg_update_time)
                st.write(f"**Last Background Update:** {bg_update_ago}")
            except:
                st.write(f"**Last Background Update:** {status['last_background_update']}")
        else:
            st.write("**Last Background Update:** Never")

def show_floating_notifications():
    """Show floating notifications that auto-dismiss."""
    notifications = get_recent_notifications(max_age_minutes=2)  # Only very recent ones
    
    if not notifications:
        return
    
    # Create floating notification area
    for notif in notifications[-2:]:  # Show only last 2
        time_ago = _get_time_ago(notif['timestamp'])
        
        # Auto-dismissing notification
        if notif['type'] == 'success':
            st.toast(f"✅ {notif['message']}", icon="✅")
        elif notif['type'] == 'warning':
            st.toast(f"⚠️ {notif['message']}", icon="⚠️")
        elif notif['type'] == 'error':
            st.toast(f"❌ {notif['message']}", icon="❌")
        else:
            st.toast(f"ℹ️ {notif['message']}", icon="ℹ️")

def _get_time_ago(timestamp):
    """Get human-readable time ago string."""
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        now = datetime.now()
        diff = now - timestamp
        
        if diff.total_seconds() < 60:
            return "just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = int(diff.total_seconds() / 86400)
            return f"{days}d ago"
    except:
        return "unknown"

def show_data_freshness_indicator():
    """Show a simple data freshness indicator."""
    status = get_update_status()
    
    if status['data_fresh']:
        st.success("🌟 Data is fresh and up-to-date")
    else:
        st.warning("🔄 Data update in progress...")
        
def create_notification_sidebar():
    """Create a comprehensive notification sidebar."""
    with st.sidebar:
        # Show background status
        show_background_status()
        
        # Show notifications if any
        st.divider()
        notifications_shown = show_notifications()
        
        if not notifications_shown:
            st.info("🔕 No recent notifications") 