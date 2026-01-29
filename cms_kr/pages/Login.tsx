import React from 'react';
import { ArrowRight } from 'lucide-react';

export const Login: React.FC<{ onLogin: () => void }> = ({ onLogin }) => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-msheireb-charcoal relative overflow-hidden">
      {/* Abstract Architectural Graphics */}
      <div className="absolute top-0 right-0 w-1/2 h-full bg-msheireb-sand opacity-5 skew-x-12 transform translate-x-20"></div>
      <div className="absolute bottom-0 left-0 w-1/3 h-1/2 bg-msheireb-gold opacity-5 -skew-x-12 transform -translate-x-20"></div>
      
      <div className="relative z-10 w-full max-w-md p-8">
        <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl">
          <div className="text-center mb-10">
            <div className="w-12 h-12 bg-msheireb-gold rounded-sm rotate-45 flex items-center justify-center mx-auto mb-6 shadow-glow">
              <div className="w-6 h-6 bg-msheireb-charcoal -rotate-45"></div>
            </div>
            <h1 className="text-3xl font-serif text-white font-bold tracking-wide">MSHEIREB</h1>
            <p className="text-msheireb-sand/60 text-sm uppercase tracking-[0.3em] mt-2">Smart Content Management</p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-xs text-msheireb-sand/50 uppercase tracking-wider mb-2">Workspace ID</label>
              <input 
                type="text" 
                defaultValue="admin@msheireb.com"
                className="w-full bg-white/10 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-msheireb-gold transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-msheireb-sand/50 uppercase tracking-wider mb-2">Access Key</label>
              <input 
                type="password" 
                defaultValue="********"
                className="w-full bg-white/10 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-msheireb-gold transition-colors"
              />
            </div>

            <button 
              onClick={onLogin}
              className="w-full bg-msheireb-gold hover:bg-msheireb-bronze text-msheireb-charcoal font-bold py-4 rounded-lg mt-6 flex items-center justify-center gap-2 transition-all hover:translate-y-[-2px] hover:shadow-lg"
            >
              <span>Enter Workspace</span>
              <ArrowRight size={18} />
            </button>
            
            <p className="text-center text-white/20 text-xs mt-6">
              Secured by Couchbase & Milvus Integration
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
