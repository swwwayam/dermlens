import React from 'react';

export default function GlassPanel({ children, className = '', elevated = false, interactive = false, style = {} }) {
  const classes = [
    'liquid-glass-card',
    elevated ? 'elevated' : '',
    interactive ? 'interactive' : '',
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={classes} style={style}>
      {children}
    </div>
  );
}
