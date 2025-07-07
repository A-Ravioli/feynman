import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, FileText, CheckCircle, AlertCircle, X } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (file: File) => Promise<void>;
  onNotification: (type: 'success' | 'error' | 'warning' | 'info', title: string, message: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, onNotification }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.name.endsWith('.phys')) {
      onNotification('error', 'Invalid File Type', 'Please upload a .phys file');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return prev;
        }
        return prev + Math.random() * 20;
      });
    }, 100);

    try {
      await onFileUpload(file);
      setUploadProgress(100);
      clearInterval(progressInterval);
      onNotification('success', 'Upload Complete', `Successfully uploaded ${file.name}`);
    } catch (error) {
      clearInterval(progressInterval);
      onNotification('error', 'Upload Failed', error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  }, [onFileUpload, onNotification]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.phys']
    },
    maxFiles: 1,
    disabled: uploading
  });

  const examples = [
    { name: 'Simple Gravity', file: 'two_body_gravity.phys', description: 'Earth-Sun gravitational system' },
    { name: 'Quantum 3D', file: 'quantum_3d.phys', description: '3D quantum particle simulation' },
    { name: 'Double Slit', file: 'double_slit.phys', description: 'Classic quantum interference experiment' },
    { name: 'Collision Demo', file: 'collision_demo.phys', description: 'Multiple particle collisions' },
  ];

  const loadExample = async (filename: string) => {
    try {
      const response = await fetch(`/examples/${filename}`);
      const content = await response.text();
      const file = new File([content], filename, { type: 'text/plain' });
      await onFileUpload(file);
    } catch (error) {
      onNotification('error', 'Failed to Load Example', 'Could not load the example file');
    }
  };

  return (
    <div className="p-6 space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-glow mb-2">Upload Physics Simulation</h1>
        <p className="text-gray-400">
          Drop your .phys file here or choose from our examples to get started
        </p>
      </motion.div>

      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300
          ${isDragActive && !isDragReject 
            ? 'border-blue-400 bg-blue-400/5 neon-glow' 
            : isDragReject 
            ? 'border-red-400 bg-red-400/5' 
            : 'border-gray-600 hover:border-blue-400 hover:bg-blue-400/5'
          }
          ${uploading ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {uploading ? (
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-blue-500 rounded-full flex items-center justify-center">
              <Upload className="w-8 h-8 text-white animate-bounce" />
            </div>
            <div className="space-y-2">
              <p className="text-lg font-semibold text-blue-400">Uploading and Processing...</p>
              <div className="w-64 mx-auto bg-gray-700 rounded-full h-2">
                <motion.div
                  className="bg-blue-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${uploadProgress}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
              <p className="text-sm text-gray-400">{Math.round(uploadProgress)}% complete</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center ${
              isDragActive 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-700 text-gray-400'
            }`}>
              {isDragReject ? (
                <X className="w-8 h-8" />
              ) : (
                <Upload className="w-8 h-8" />
              )}
            </div>
            
            <div>
              {isDragActive ? (
                isDragReject ? (
                  <p className="text-lg font-semibold text-red-400">
                    Invalid file type. Please upload a .phys file.
                  </p>
                ) : (
                  <p className="text-lg font-semibold text-blue-400">
                    Drop your .phys file here!
                  </p>
                )
              ) : (
                <div className="space-y-2">
                  <p className="text-lg font-semibold text-gray-300">
                    Drag & drop your .phys file here
                  </p>
                  <p className="text-gray-400">or click to browse files</p>
                </div>
              )}
            </div>
          </div>
        )}
      </motion.div>

      {/* Examples Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="space-y-4"
      >
        <h2 className="text-xl font-semibold text-glow">Example Simulations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {examples.map((example, index) => (
            <motion.button
              key={example.file}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
              onClick={() => loadExample(example.file)}
              disabled={uploading}
              className="card text-left p-4 hover:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-start space-x-3">
                <div className="p-2 bg-blue-500/20 rounded-lg flex-shrink-0">
                  <FileText className="w-5 h-5 text-blue-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-white truncate">{example.name}</h3>
                  <p className="text-sm text-gray-400 mt-1">{example.description}</p>
                  <div className="flex items-center space-x-2 mt-2">
                    <span className="text-xs px-2 py-1 bg-gray-700 rounded text-gray-300">
                      {example.file}
                    </span>
                  </div>
                </div>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Help Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card p-6"
      >
        <h3 className="text-lg font-semibold text-glow mb-4">Need Help?</h3>
        <div className="space-y-3 text-sm text-gray-300">
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Upload .phys files containing PhysicsLang simulation code</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Supports both classical and quantum physics simulations</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Real-time validation and error reporting</span>
          </div>
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
            <span>Maximum file size: 10MB</span>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default FileUpload;