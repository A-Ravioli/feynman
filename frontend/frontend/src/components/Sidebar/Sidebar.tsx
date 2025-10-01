import React from 'react';
import { motion } from 'framer-motion';
import { 
  ChevronLeft, 
  ChevronRight,
  BarChart3,
  Globe,
  Atom,
  Waves,
  Upload,
  Settings,
  Zap
} from 'lucide-react';
import type { NavigationItem } from '../../types';

interface SidebarProps {
  items: NavigationItem[];
  collapsed: boolean;
  onItemClick: (itemId: string) => void;
  onToggle: () => void;
}

const iconMap = {
  BarChart3: BarChart3,
  Globe: Globe,
  Atom: Atom,
  Waves: Waves,
  Upload: Upload,
  Settings: Settings,
  Zap: Zap,
};

const Sidebar: React.FC<SidebarProps> = ({ items, collapsed, onItemClick, onToggle }) => {
  const sidebarVariants = {
    expanded: {
      width: 256
    },
    collapsed: {
      width: 64
    }
  };

  const contentVariants = {
    expanded: {
      opacity: 1,
      x: 0,
      transition: {
        delay: 0.1,
        duration: 0.2
      }
    },
    collapsed: {
      opacity: 0,
      x: -20,
      transition: {
        duration: 0.1
      }
    }
  };

  return (
    <motion.aside
      variants={sidebarVariants}
      animate={collapsed ? 'collapsed' : 'expanded'}
      className="fixed left-0 top-0 h-full bg-dark-surface glass-effect border-r border-gray-700 z-20 flex flex-col"
    >
      {/* Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-gray-700">
        {!collapsed && (
          <motion.div
            variants={contentVariants}
            animate={collapsed ? 'collapsed' : 'expanded'}
            className="flex items-center space-x-3"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-semibold text-glow">Feynman</span>
          </motion.div>
        )}
        
        <button
          onClick={onToggle}
          className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors duration-200 neon-glow"
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5 text-gray-300" />
          ) : (
            <ChevronLeft className="w-5 h-5 text-gray-300" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-2 overflow-y-auto">
        {items.map((item, index) => {
          const Icon = iconMap[item.icon as keyof typeof iconMap] || BarChart3;
          
          return (
            <motion.button
              key={item.id}
              onClick={() => onItemClick(item.id)}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`
                w-full flex items-center space-x-3 px-3 py-3 rounded-xl transition-all duration-300 group
                ${item.active 
                  ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/50 text-white neon-glow' 
                  : 'hover:bg-gray-700/50 text-gray-300 hover:text-white hover:border-gray-600 border border-transparent'
                }
              `}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className={`
                relative flex-shrink-0
                ${item.active ? 'text-blue-400' : 'text-gray-400 group-hover:text-blue-400'}
              `}>
                <Icon className="w-6 h-6 transition-colors duration-200" />
                {item.active && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="absolute inset-0 w-6 h-6"
                  >
                    <div className="w-full h-full bg-blue-400/20 rounded-lg animate-pulse" />
                  </motion.div>
                )}
              </div>
              
              {!collapsed && (
                <motion.div
                  variants={contentVariants}
                  animate={collapsed ? 'collapsed' : 'expanded'}
                  className="flex-1 text-left"
                >
                  <span className="text-sm font-medium">{item.label}</span>
                  {item.badge && (
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="ml-2 px-2 py-1 text-xs bg-red-500 text-white rounded-full"
                    >
                      {item.badge}
                    </motion.span>
                  )}
                </motion.div>
              )}
            </motion.button>
          );
        })}
      </nav>

      {/* Status Footer */}
      <div className="p-4 border-t border-gray-700">
        {!collapsed ? (
          <motion.div
            variants={contentVariants}
            animate={collapsed ? 'collapsed' : 'expanded'}
            className="space-y-2"
          >
            <div className="flex items-center space-x-2 text-xs text-gray-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span>System Online</span>
            </div>
            <div className="text-xs text-gray-500 font-mono">
              v1.0.0-beta
            </div>
          </motion.div>
        ) : (
          <div className="flex justify-center">
            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" />
          </div>
        )}
      </div>
    </motion.aside>
  );
};

export default Sidebar;