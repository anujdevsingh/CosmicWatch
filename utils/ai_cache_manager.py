#!/usr/bin/env python3
"""
AI Cache Manager for Space Debris Dashboard
Intelligent caching of AI predictions to improve performance
"""

from datetime import datetime, timedelta
from utils.database import get_db, SpaceDebris, set_metadata_value, get_metadata_value

class AICacheManager:
    """Manages AI prediction caching and smart re-analysis."""
    
    def __init__(self):
        self.confidence_threshold = 0.8  # Re-analyze if confidence < 80%
        self.max_cache_age_hours = 24    # Re-analyze if prediction > 24 hours old
        self.significant_change_threshold = {
            'altitude': 10.0,    # 10km altitude change
            'velocity': 0.5,     # 0.5 km/s velocity change
            'size': 0.2          # 0.2m size change
        }
    
    def should_reanalyze(self, debris_obj):
        """Determine if an object needs AI re-analysis."""
        # Check if object has cached AI prediction
        if not debris_obj.ai_risk_level or not debris_obj.ai_last_predicted:
            return True, "No cached prediction"
        
        # Check prediction age
        age_hours = (datetime.now() - debris_obj.ai_last_predicted).total_seconds() / 3600
        if age_hours > self.max_cache_age_hours:
            return True, f"Prediction too old ({age_hours:.1f}h)"
        
        # Check confidence level
        if debris_obj.ai_confidence and debris_obj.ai_confidence < self.confidence_threshold:
            return True, f"Low confidence ({debris_obj.ai_confidence:.2f})"
        
        # Check for significant changes (would need change tracking)
        # For now, we'll assume data from same source is consistent
        
        return False, "Using cached prediction"
    
    def get_cached_prediction(self, debris_obj):
        """Get cached AI prediction for an object."""
        if not debris_obj.ai_risk_level:
            return None
            
        return {
            'risk_level': debris_obj.ai_risk_level,
            'confidence': debris_obj.ai_confidence or 0.0,
            'enhanced': bool(debris_obj.ai_enhanced),
            'last_predicted': debris_obj.ai_last_predicted,
            'probabilities': {}  # Could be stored as JSON in future
        }
    
    def cache_prediction(self, debris_id, prediction):
        """Cache an AI prediction for future use."""
        try:
            db = next(get_db())
            debris_obj = db.query(SpaceDebris).filter(SpaceDebris.id == debris_id).first()
            
            if debris_obj:
                debris_obj.ai_risk_level = prediction.get('risk_level', 'UNKNOWN')
                debris_obj.ai_confidence = prediction.get('confidence', 0.0)
                debris_obj.ai_last_predicted = datetime.now()
                debris_obj.ai_enhanced = 1 if prediction.get('enhanced', False) else 0
                
                db.commit()
                return True
        except Exception as e:
            print(f"⚠️ Error caching prediction for {debris_id}: {e}")
            return False
    
    def get_cache_statistics(self):
        """Get statistics about AI cache usage."""
        try:
            db = next(get_db())
            
            total_objects = db.query(SpaceDebris).count()
            cached_objects = db.query(SpaceDebris).filter(
                SpaceDebris.ai_risk_level.isnot(None)
            ).count()
            
            # Recent predictions (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_predictions = db.query(SpaceDebris).filter(
                SpaceDebris.ai_last_predicted > recent_cutoff
            ).count()
            
            # High confidence predictions
            high_confidence = db.query(SpaceDebris).filter(
                SpaceDebris.ai_confidence >= self.confidence_threshold
            ).count()
            
            return {
                'total_objects': total_objects,
                'cached_objects': cached_objects,
                'cache_percentage': (cached_objects / total_objects * 100) if total_objects > 0 else 0,
                'recent_predictions': recent_predictions,
                'high_confidence': high_confidence,
                'confidence_threshold': self.confidence_threshold,
                'max_cache_age_hours': self.max_cache_age_hours
            }
            
        except Exception as e:
            print(f"⚠️ Error getting cache statistics: {e}")
            return {
                'total_objects': 0,
                'cached_objects': 0,
                'cache_percentage': 0,
                'recent_predictions': 0,
                'high_confidence': 0,
                'confidence_threshold': self.confidence_threshold,
                'max_cache_age_hours': self.max_cache_age_hours
            }
    
    def optimize_cache(self):
        """Optimize the AI cache by cleaning old/invalid entries."""
        try:
            db = next(get_db())
            
            # Remove very old predictions (>7 days)
            old_cutoff = datetime.now() - timedelta(days=7)
            old_count = db.query(SpaceDebris).filter(
                SpaceDebris.ai_last_predicted < old_cutoff
            ).update({
                SpaceDebris.ai_risk_level: None,
                SpaceDebris.ai_confidence: None,
                SpaceDebris.ai_last_predicted: None,
                SpaceDebris.ai_enhanced: 0
            })
            
            db.commit()
            print(f"🧹 Cleaned {old_count} old AI cache entries")
            
            # Update metadata
            set_metadata_value('last_cache_optimization', datetime.now().isoformat())
            
            return old_count
            
        except Exception as e:
            print(f"⚠️ Error optimizing cache: {e}")
            return 0

# Global instance
_ai_cache_manager = None

def get_ai_cache_manager():
    """Get the global AI cache manager."""
    global _ai_cache_manager
    if _ai_cache_manager is None:
        _ai_cache_manager = AICacheManager()
    return _ai_cache_manager

def should_reanalyze_object(debris_obj):
    """Check if an object needs AI re-analysis."""
    manager = get_ai_cache_manager()
    return manager.should_reanalyze(debris_obj)

def get_cached_ai_prediction(debris_obj):
    """Get cached AI prediction for an object."""
    manager = get_ai_cache_manager()
    return manager.get_cached_prediction(debris_obj)

def cache_ai_prediction(debris_id, prediction):
    """Cache an AI prediction."""
    manager = get_ai_cache_manager()
    return manager.cache_prediction(debris_id, prediction)

def get_ai_cache_stats():
    """Get AI cache statistics."""
    manager = get_ai_cache_manager()
    return manager.get_cache_statistics()

def optimize_ai_cache():
    """Optimize the AI cache."""
    manager = get_ai_cache_manager()
    return manager.optimize_cache() 