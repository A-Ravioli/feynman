import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import type { NotificationMessage } from '../../types';

interface NotificationCenterProps {
  notifications: NotificationMessage[];
  onRemove: (id: string) => void;
}

const iconMap = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const colorMap = {
  success: 'from-green-500 to-emerald-500',
  error: 'from-red-500 to-pink-500',
  warning: 'from-yellow-500 to-orange-500',
  info: 'from-blue-500 to-cyan-500',
};

const NotificationCenter: React.FC<NotificationCenterProps> = ({ notifications, onRemove }) => {
  // Auto-remove notifications
  useEffect(() => {
    notifications.forEach(notification => {
      if (notification.autoClose && notification.duration) {
        const timer = setTimeout(() => {
          onRemove(notification.id);
        }, notification.duration);

        return () => clearTimeout(timer);
      }
    });
  }, [notifications, onRemove]);

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      <AnimatePresence>
        {notifications.map((notification) => {
          const Icon = iconMap[notification.type];
          const colorClass = colorMap[notification.type];

          return (
            <motion.div
              key={notification.id}
              initial={{ opacity: 0, x: 300, scale: 0.8 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 300, scale: 0.8 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              className="w-80 glass-effect border border-gray-700 rounded-xl p-4 shadow-xl"
            >
              <div className="flex items-start space-x-3">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${colorClass} flex-shrink-0`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-white text-sm">
                    {notification.title}
                  </h4>
                  <p className="text-gray-300 text-sm mt-1 leading-relaxed">
                    {notification.message}
                  </p>
                  <div className="text-xs text-gray-500 mt-2">
                    {notification.timestamp.toLocaleTimeString()}
                  </div>
                </div>
                
                <button
                  onClick={() => onRemove(notification.id)}
                  className="flex-shrink-0 p-1 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <X className="w-4 h-4 text-gray-400" />
                </button>
              </div>
              
              {/* Progress bar for auto-close */}
              {notification.autoClose && notification.duration && (
                <motion.div
                  className={`h-1 bg-gradient-to-r ${colorClass} rounded-full mt-3`}
                  initial={{ width: '100%' }}
                  animate={{ width: '0%' }}
                  transition={{ duration: notification.duration / 1000, ease: 'linear' }}
                />
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
};

export default NotificationCenter;