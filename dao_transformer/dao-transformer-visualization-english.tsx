import React, { useState } from 'react';
import { ArrowDown, Zap, Anchor, Scissors, Droplets } from 'lucide-react';

// Five Elements color settings
const elementColors = {
  wood: '#4CAF50',  // Wood-Green
  fire: '#F44336',  // Fire-Red
  earth: '#FFC107',  // Earth-Yellow
  metal: '#9E9E9E',  // Metal-Silver Gray
  water: '#2196F3',  // Water-Blue
  neutral: '#6d4c41', // Neutral-Brown
};

// Custom Tree icon component (replacing Tree)
const TreeIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v18"/>
    <path d="M8 6l4 -3l4 3"/>
    <path d="M8 12l4 -3l4 3"/>
    <path d="M8 18l4 -3l4 3"/>
  </svg>
);

// Mechanism point strength indicator
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

// Five Elements component
const ElementComponent = ({ element, active, onClick }) => {
  const elements = {
    wood: { icon: <TreeIcon size={24} />, name: "Wood", desc: "Expansion" },
    fire: { icon: <Zap size={24} />, name: "Fire", desc: "Acceleration" },
    earth: { icon: <Anchor size={24} />, name: "Earth", desc: "Stabilization" },
    metal: { icon: <Scissors size={24} />, name: "Metal", desc: "Refinement" },
    water: { icon: <Droplets size={24} />, name: "Water", desc: "Adaptation" },
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

// Transformation flow diagram
const TransformationFlow = () => {
  const [activeElement, setActiveElement] = useState('wood');
  const [mechanismStrength, setMechanismStrength] = useState(0.6);
  
  // Simulate effects of different elements
  const handleElementClick = (element) => {
    setActiveElement(element);
    
    // Simulate mechanism point strength changes
    switch(element) {
      case 'wood': setMechanismStrength(0.6); break;
      case 'fire': setMechanismStrength(0.8); break;
      case 'earth': setMechanismStrength(0.5); break;
      case 'metal': setMechanismStrength(0.7); break;
      case 'water': setMechanismStrength(0.4); break;
      default: setMechanismStrength(0.5);
    }
  };
  
  // Generate output description based on active element
  const getOutputDescription = () => {
    switch(activeElement) {
      case 'wood':
        return "Expands hidden space, enhances creative expression, promotes thought expansion";
      case 'fire':
        return "Accelerates state transitions, amplifies differentiating features, increases attention";
      case 'earth':
        return "Stabilizes system states, balances elemental relationships, forms harmonious whole";
      case 'metal':
        return "Refines key information, extracts core content, enhances precision";
      case 'water':
        return "Flexibly adapts to context, flows naturally, creates subtle changes";
      default:
        return "Transformation in progress...";
    }
  };
  
  return (
    <div className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-xl">
      <h2 className="text-2xl font-bold mb-8" style={{ color: elementColors.neutral }}>Mechanism Point Transformation Flow</h2>
      
      {/* Input layer */}
      <div className="bg-white p-4 rounded-lg shadow-md w-full text-center mb-4">
        <div className="text-sm text-gray-500 mb-1">Input</div>
        <div className="font-mono">Hidden State Vector (h<sub>t</sub>)</div>
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* Mechanism point detection */}
      <div className="flex items-center justify-center bg-violet-100 p-4 rounded-lg shadow-md w-full mb-4">
        <div className="flex-1">
          <div className="text-sm text-gray-500 mb-1">Mechanism Point Detection</div>
          <div className="font-mono">Detect key influence nodes and state transformation patterns</div>
        </div>
        <MechanismPointIndicator strength={mechanismStrength} />
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* Five Elements transformation layer */}
      <div className="bg-gray-100 p-4 rounded-lg shadow-md w-full mb-4">
        <div className="text-sm text-gray-500 mb-1 text-center">Five Elements Transformation</div>
        <div className="grid grid-cols-5 gap-2 my-2">
          <ElementComponent element="wood" active={activeElement === 'wood'} onClick={() => handleElementClick('wood')} />
          <ElementComponent element="fire" active={activeElement === 'fire'} onClick={() => handleElementClick('fire')} />
          <ElementComponent element="earth" active={activeElement === 'earth'} onClick={() => handleElementClick('earth')} />
          <ElementComponent element="metal" active={activeElement === 'metal'} onClick={() => handleElementClick('metal')} />
          <ElementComponent element="water" active={activeElement === 'water'} onClick={() => handleElementClick('water')} />
        </div>
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* Position-timing unity layer */}
      <div className="bg-indigo-100 p-4 rounded-lg shadow-md w-full mb-4">
        <div className="text-sm text-gray-500 mb-1 text-center">Position-Timing Unity</div>
        <div className="flex justify-center space-x-4">
          <div className="bg-white p-2 rounded border border-indigo-200">Position</div>
          <div className="text-xl">⟡</div>
          <div className="bg-white p-2 rounded border border-indigo-200">Timing</div>
        </div>
      </div>
      
      <ArrowDown className="text-gray-400 my-2" />
      
      {/* Output layer */}
      <div className="bg-white p-4 rounded-lg shadow-md w-full text-center">
        <div className="text-sm text-gray-500 mb-1">Output</div>
        <div className="font-mono">Transformed Vector (h<sub>t+1</sub>)</div>
        <div className="mt-2 p-2 rounded" style={{ backgroundColor: `${elementColors[activeElement]}10`, color: elementColors[activeElement] }}>
          {getOutputDescription()}
        </div>
      </div>
    </div>
  );
};

// Main application component
const DaoTransformerApp = () => {
  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6" style={{ color: elementColors.neutral }}>Mechanism Model - MechanismPointsLLM</h1>
      
      <div className="max-w-3xl w-full">
        <TransformationFlow />
        
        <div className="mt-8 p-6 bg-white rounded-xl shadow-md">
          <h2 className="text-xl font-bold mb-4" style={{ color: elementColors.neutral }}>Mechanism Model Features</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.wood}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.wood }}>Mechanism Point Perception</h3>
              <p className="text-sm">Identifies key influence nodes in systems, creating large impacts from small inputs</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.fire}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.fire }}>Five Elements Transformation</h3>
              <p className="text-sm">Wood, Fire, Earth, Metal, Water transformation patterns adapting to different text characteristics</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.earth}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.earth }}>Position-Timing Unity</h3>
              <p className="text-sm">Integrates "where" and "when", capturing optimal intervention timing</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.metal}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.metal }}>Refined Attention</h3>
              <p className="text-sm">Optimizes attention allocation through mechanism awareness, extracting core information</p>
            </div>
            
            <div className="p-3 rounded-lg" style={{ backgroundColor: `${elementColors.water}10` }}>
              <h3 className="font-bold" style={{ color: elementColors.water }}>Flow Adaptation</h3>
              <p className="text-sm">Self-adaptive training and generation, following hidden dynamics</p>
            </div>
            
            <div className="p-3 rounded-lg bg-purple-50">
              <h3 className="font-bold text-purple-700">Action Through Non-Action</h3>
              <p className="text-sm">Following natural principles to achieve optimal results with minimal intervention</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6 text-center text-sm text-gray-500">
        Heaven-Human Unity · Five Elements & Eight Trigrams · Mechanism Point Transformation · Position-Timing Unity
      </div>
    </div>
  );
};

export default DaoTransformerApp;