import React, { useEffect, useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { Database, HardDrive, Cpu, Activity } from 'lucide-react';
import { ApiService, DashboardData } from '../services/apiService';

export const Dashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);

  useEffect(() => {
    ApiService.getDashboardStats().then(setData);
  }, []);

  // Fallback data if API fails or is loading empty
  const activityData = data?.activity || [];
  const pieData = data?.distribution || [];
  const stats = data?.stats;

  const COLORS = ['#2D3748', '#C5A065', '#8C7353', '#E6E0D4', '#D4AF37'];

  // Helper to format bytes
  const formatSize = (bytes: number = 0) => {
    if (bytes === 0) return '0 MB';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          title="Total Documents" 
          value={stats?.totalDocuments || 0} 
          icon={<Database className="text-msheireb-gold" />} 
          subtitle="Indexed in MongoDB"
        />
        <StatCard 
          title="Vector Chunks" 
          value={stats?.totalChunks || 0} 
          icon={<Cpu className="text-blue-500" />} 
          subtitle="Stored in Milvus"
        />
        <StatCard 
          title="Storage Used" 
          value={formatSize(stats?.storageUsed)} 
          icon={<HardDrive className="text-green-600" />} 
          subtitle="Total file size"
        />
        <StatCard 
          title="System Health" 
          value="Online" 
          icon={<Activity className="text-green-500" />} 
          subtitle="API & DBs Connected"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm border border-msheireb-sand">
          <h3 className="text-lg font-serif font-semibold text-msheireb-charcoal mb-6">Ingestion Activity (7 Days)</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activityData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E6E0D4" />
                <XAxis dataKey="name" tick={{fill: '#8C7353'}} axisLine={false} tickLine={false} />
                <YAxis tick={{fill: '#8C7353'}} axisLine={false} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#2D3748', border: 'none', borderRadius: '8px', color: '#fff' }}
                  cursor={{ fill: '#F5F5F0' }}
                />
                <Bar dataKey="vectors" name="Vectors" fill="#C5A065" radius={[4, 4, 0, 0]} />
                <Bar dataKey="docs" name="Documents" fill="#2D3748" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-msheireb-sand">
          <h3 className="text-lg font-serif font-semibold text-msheireb-charcoal mb-6">Content Distribution</h3>
          <div className="h-80 w-full flex items-center justify-center">
             {pieData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      fill="#8884d8"
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
             ) : (
                 <div className="text-gray-400 text-sm">No data available</div>
             )}
          </div>
          <div className="flex flex-wrap justify-center gap-4 mt-4 text-sm text-gray-500">
            {pieData.map((d, i) => (
              <div key={i} className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full" style={{backgroundColor: COLORS[i % COLORS.length]}}></div>
                {d.name} ({d.value})
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ title, value, icon, subtitle }: { title: string, value: string | number, icon: any, subtitle: string }) => (
  <div className="bg-white p-6 rounded-xl shadow-sm border border-msheireb-sand hover:shadow-md transition-shadow group">
    <div className="flex justify-between items-start">
      <div>
        <p className="text-sm font-medium text-msheireb-charcoal/60 uppercase tracking-wide">{title}</p>
        <h3 className="text-3xl font-bold text-msheireb-charcoal mt-2 group-hover:text-msheireb-gold transition-colors">{value}</h3>
      </div>
      <div className="p-3 bg-msheireb-limestone rounded-lg group-hover:scale-110 transition-transform duration-300">
        {icon}
      </div>
    </div>
    <p className="text-xs text-gray-400 mt-4 font-light">{subtitle}</p>
  </div>
);