import React, { useState } from 'react';
import { ArrowDown, Zap, Anchor, Scissors, Droplets } from 'lucide-react';

// 五行顏色設定
const elementColors = {
  wood: '#4CAF50',  // 木-綠
  fire: '#F44336',  // 火-紅
  earth: '#FFC107',  // 土-黃
  metal: '#9E9E9E',  // 金-銀灰
  water: '#2196F3',  // 水-藍
  neutral: '#6d4c41', // 中性-褐色
};

// 自定義樹圖標組件 (替代Tree)
const TreeIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v18"/>
    <path d="M8 6l4 -3l4 3"/>
    <path d="M8 12l4 -3l4 3"/>
    <path d="M8 18l4 -3l4 3"/>
  </svg>
);

// 機制點強度指示器
const MechanismPointIndicator = ({ strength }) => {
  const height = `${Math.max(5, Math.min(100, strength * 100))}%`;
  return (
    <div className="w-4 h-16 bg-gray-200 rounded-full overflow-hidden">
      <div 
        className="bg-purple-600 w-full rounded-full transition-all duration-500 ease-in-out"
        style={{ height: height, marginTop: `calc(100% - ${height})` }}
      />
    </div>
  );
};

// 五行元素組件
const ElementComponent = ({ element, active, onClick }) => {
  const elements = {
    wood: { icon: <TreeIcon size={24} />, name: "木 Wood", desc: "擴展" },
    fire: { icon: <Zap size={24} />, name: "火 Fire", desc: "加速" },
    earth: { icon: <Anchor size={24} />, name: "土 Earth", desc: "穩定" },
    metal: { icon: <Scissors size={24} />, name: "金 Metal", desc: "精煉" },
    water: { icon: <Droplets size={24} />, name: "水 Water", desc: "適應" },
  };
  
  const { icon, name, desc } = elements[element];
  
  return (
    <div 
      className={`flex flex-col items-center justify-center p-4 rounded-lg cursor-pointer transition-all duration-300 transform ${active ? 'scale-110 shadow-lg' : 'opacity-70'}`}
      style={{ backgroundColor: `${elementColors[element]}20`, borderColor: elementColors[element], borderWidth: active ? '2px' : '1px' }}
      onClick={onClick}
    >
      <div className="text-xl mb-1" style={{ color: elementColors[element] }}>
        {icon}
      </div>
      <div className="font-bold" style={{ color: elementColors[element] }}>{name}</div>
      <div className="text-xs text-gray-600">{desc}</div>
    </div>
  );
};

// 轉化流程圖
const TransformationFlow = () => {
  const [activeElement, setActiveElement] = useState('wood');
  const [mechanismStrength, setMechanismStrength] = useState(0.6);
  
  // 模擬不同元素的效果
  const handleElementClick = (element) => {
    setActiveElement(element);
    
    // 模擬機制點強度變化
    switch(element) {
      case 'wood': setMechanismStrength(0.6); break;
      case 'fire': setMechanismStrength(0.8); break;
      case 'earth': setMechanismStrength(0.5); break;
      case 'metal': setMechanismStrength(0.7); break;
      case 'water': setMechanismStrength(0.4); break;
      default: setMechanismStrength(0.5);
    }
  };
  
  // 根據活躍元素生成對應的輸出描述
  const getOutputDescription = () => {
    switch(activeElement) {
      case 'wood':
        return "擴展隱藏空間，增強創造性表達，促進思維拓展";
      case 'fire':
        return "加速狀態轉換，放大差異性特徵，增強關注度";
      case 'earth':
        return "穩定系統狀態，平衡元素關係，形成協調整體";
      case 'metal':
        return "精煉關鍵信息，提取核心內容，增強精確度";
      case 'water':
        return "靈活適應上下文，順勢流轉，無形改變";
      default:
        return "轉化進行中...";
    }
  };
  
  return (
    <div className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-xl">
      <h2 className="text-2xl font-bold mb-8" style={{ color: elementColors.neutral }}>機制點轉化流程</h2>
      
      {/* 輸入層 */}
      <div className="bg-white p-4 rounded-lg shadow-md w-full text-center mb-4">
        <div className="text-sm text-gray-500 mb-1">輸入 (Input)</div>
        <div className="font-mono">隱藏狀態向量 (h<sub>t</sub>)</div>
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* 機制點檢測 */}
      <div className="flex items-center justify-center bg-violet-100 p-4 rounded-lg shadow-md w-full mb-4">
        <div className="flex-1">
          <div className="text-sm text-gray-500 mb-1">機制點檢測 (Mechanism Point Detection)</div>
          <div className="font-mono">檢測關鍵影響節點與狀態轉化模式</div>
        </div>
        <MechanismPointIndicator strength={mechanismStrength} />
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* 五行轉化層 */}
      <div className="bg-gray-100 p-4 rounded-lg shadow-md w-full mb-4">
        <div className="text-sm text-gray-500 mb-1 text-center">五行轉化 (Five Elements Transformation)</div>
        <div className="grid grid-cols-5 gap-2 my-2">
          <ElementComponent element="wood" active={activeElement === 'wood'} onClick={() => handleElementClick('wood')} />
          <ElementComponent element="fire" active={activeElement === 'fire'} onClick={() => handleElementClick('fire')} />
          <ElementComponent element="earth" active={activeElement === 'earth'} onClick={() => handleElementClick('earth')} />
          <ElementComponent element="metal" active={activeElement === 'metal'} onClick={() => handleElementClick('metal')} />
          <ElementComponent element="water" active={activeElement === 'water'} onClick={() => handleElementClick('water')} />
        </div>
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* 位置時機統一層 */}
      <div className="bg-indigo-100 p-4 rounded-lg shadow-md w-full mb-4">
        <div className="text-sm text-gray-500 mb-1 text-center">位置時機統一 (Position-Timing Unity)</div>
        <div className="flex justify-center space-x-4">
          <div className="bg-white p-2 rounded border border-indigo-200">位置編碼 (Position)</div>
          <div className="text-xl">⟡</div>
          <div className="bg-white p-2 rounded border border-indigo-200">時序關係 (Timing)</div>
        </div>
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* 輸出層 */}
      <div className="bg-white p-4 rounded-lg shadow-md w-full text-center">
        <div className="text-sm text-gray-500 mb-1">輸出 (Output)</div>
        <div className="font-mono">轉化後向量 (h<sub>t+1</sub>)</div>
        <div className="mt-2 p-2 rounded" style={{ backgroundColor: `${elementColors[activeElement]}10`, color: elementColors[activeElement] }}>
          {getOutputDescription()}
        </div>
      </div>
    </div>
  );
};

// 主應用組件
const DaoTransformerApp = () => {
  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6" style={{ color: elementColors.neutral }}>玄機模型 - MechanismPointsLLM</h1>
      
      <div className="max-w-3xl w-full">
        <TransformationFlow />
        
        <div className="mt-8 p-6 bg-white rounded-xl shadow-md">
          <h2 className="text-xl font-bold mb-4" style={{ color: elementColors.neutral }}>玄機模型特性</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.wood}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.wood }}>機制點感知</h3>
              <p className="text-sm">識別系統中的關鍵影響節點，小投入創造大影響</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.fire}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.fire }}>五行轉化</h3>
              <p className="text-sm">木火土金水五種轉化模式，適應不同文本特性</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.earth}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.earth }}>位置時機統一</h3>
              <p className="text-sm">整合「何處」與「何時」，把握最佳干預時機</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.metal}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.metal }}>精煉注意力</h3>
              <p className="text-sm">通過機制感知優化注意力分配，提取核心信息</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.water}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.water }}>流動適應</h3>
              <p className="text-sm">自適應訓練與生成，順應隱藏動態</p>
            </div>
            
            <div className="p-3 rounded-lg bg-purple-50">
              <h3 className="font-bold text-purple-700">無為而生</h3>
              <p className="text-sm">遵循自然規律，以最小干預達成最佳效果</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6 text-center text-sm text-gray-500">
        天人合一 · 五行八卦 · 機制點轉化 · 位置時機統一
      </div>
    </div>
  );
};

export default DaoTransformerApp;