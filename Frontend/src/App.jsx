import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence, useSpring, useTransform } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Zap, Activity, ScanFace, CheckCircle2, Terminal, Sun, Moon } from 'lucide-react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import axios from 'axios';

/* Imports for live charts and tilt effects */
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import Tilt from 'react-parallax-tilt';

const BACKEND_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/video";

/* ============================================================
   FLOATING PARTICLES — Ambient background particles using
   CSS var(--particle-color) for dark/light mode theming.
   ============================================================ */
function FloatingParticles() {
  /* Generate 60 particles with randomized position, delay, duration, and size */
  const particles = useMemo(() =>
    Array.from({ length: 60 }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      delay: Math.random() * 8,
      duration: 6 + Math.random() * 8,
      size: 3 + Math.random() * 4,
    }))
    , []);

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {particles.map(p => (
        <div
          key={p.id}
          className="absolute rounded-full"
          style={{
            left: `${p.left}%`,
            bottom: '-5%',
            width: `${p.size}px`,
            height: `${p.size}px`,
            backgroundColor: 'var(--particle-color)',
            animation: `particle-drift ${p.duration}s linear ${p.delay}s infinite`,
          }}
        />
      ))}
    </div>
  );
}


function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [socket, setSocket] = useState(null);
  const [displayImage, setDisplayImage] = useState(null);

  const [appState, setAppState] = useState("uncalibrated");
  const [calibProgress, setCalibProgress] = useState(0);

  const [trustScore, setTrustScore] = useState(100);
  const [verdict, setVerdict] = useState("REAL");
  const [lipProb, setLipProb] = useState(0);
  const [eyeProb, setEyeProb] = useState(0);
  const [isInjecting, setIsInjecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  /* Chart history buffer for real-time ECG-style anomaly graph */
  const [chartData, setChartData] = useState([]);
  /* Glitch trigger state for verdict change animation */
  const [glitchActive, setGlitchActive] = useState(false);
  /* Track previous verdict to detect changes */
  const prevVerdictRef = useRef("REAL");
  /* Dark/Light mode state — defaults to dark */
  const [isDarkMode, setIsDarkMode] = useState(true);
  /* Real-time calibration elapsed seconds counter */
  const [calibElapsed, setCalibElapsed] = useState(0);

  /* Apply data-theme attribute to html element when mode changes */
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(e => console.error("Play Error:", e));
        };
      }
    } catch (err) {
      console.error("Camera Error:", err);
      alert("Camera failed to start. Please check permissions!");
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDisplayImage(data.image);

      if (data.metrics) {
        setAppState(prevState => {
          if (data.metrics.status === "enrolling" && prevState !== "calibrating") return "calibrating";
          if (data.metrics.status === "enrolled" && prevState === "calibrating") return "calibration_complete";
          return prevState;
        });

        if (data.metrics.status === "enrolling") setCalibProgress(data.metrics.progress);

        if (data.metrics.status === "active" || data.metrics.status === "enrolled") {
          setTrustScore(Math.round(data.metrics.trust_score * 100));
          setVerdict(data.metrics.verdict);
          setLipProb(data.metrics.lip_prob_fake);
          setEyeProb(data.metrics.eye_prob_fake);

          /* Push new data point into the rolling chart history (max 60 points) */
          setChartData(prev => {
            const next = [...prev, { lip: data.metrics.lip_prob_fake, eye: data.metrics.eye_prob_fake }];
            return next.length > 60 ? next.slice(-60) : next;
          });
        }
      }
    };

    setSocket(ws);
    startWebcam();
    return () => {
      ws.close();
      stopWebcam();
    }
  }, [startWebcam, stopWebcam]);

  /* Detect verdict changes and trigger the glitch animation */
  useEffect(() => {
    if (verdict !== prevVerdictRef.current) {
      setGlitchActive(true);
      const timeout = setTimeout(() => setGlitchActive(false), 400);
      prevVerdictRef.current = verdict;
      return () => clearTimeout(timeout);
    }
  }, [verdict]);

  /* Real-time calibration elapsed seconds counter */
  useEffect(() => {
    if (appState === "calibrating") {
      setCalibElapsed(0);
      const timer = setInterval(() => {
        setCalibElapsed(prev => prev + 1);
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [appState]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (socket && socket.readyState === WebSocket.OPEN && canvasRef.current) {
        if (isInjecting) {
          const frame = canvasRef.current.toDataURL('image/jpeg', 0.1);
          socket.send(frame);
        } else if (videoRef.current && videoRef.current.readyState === 4) {
          const ctx = canvasRef.current.getContext('2d');
          ctx.drawImage(videoRef.current, 0, 0, 640, 480);
          const frame = canvasRef.current.toDataURL('image/jpeg', 0.7);
          socket.send(frame);
        }
      }
    }, 33);
    return () => clearInterval(interval);
  }, [socket, isInjecting]);

  const startCalibration = async () => {
    try {
      await axios.post(`${BACKEND_URL}/api/enroll`);
      setAppState("calibrating");
    } catch (err) {
      console.error("API Error", err);
    }
  };

  const toggleInjection = async () => {
    try {
      const newState = !isInjecting;
      await axios.post(`${BACKEND_URL}/api/toggle-injection`, { active: newState });
      setIsInjecting(newState);

      if (newState) stopWebcam();
      else startWebcam();

    } catch (err) {
      console.error("API Error", err);
    }
  };

  const isSafe = verdict === "REAL";
  const themeColor = isSafe ? "#10B981" : "#e11d48";
  const secondsRemaining = calibElapsed;

  /* ============================================================
     BOOT-UP SEQUENCE — Each section has explicit delay-based
     individual animations cascading independently.
     ============================================================ */
  const headerAnim = {
    initial: { opacity: 0, y: -40 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.7, delay: 0.1, ease: "easeOut" }
  };
  const videoPanelAnim = {
    initial: { opacity: 0, scale: 0.9, filter: "blur(10px)" },
    animate: { opacity: 1, scale: 1, filter: "blur(0px)" },
    transition: { duration: 0.8, delay: 0.4, ease: "easeOut" }
  };
  const sidePanelAnim = {
    initial: { opacity: 0, x: 60 },
    animate: { opacity: 1, x: 0 },
    transition: { duration: 0.6, delay: 0.7, ease: "easeOut" }
  };

  /* Button interaction variants for micro-animations */
  const buttonTap = { scale: 0.97 };
  const buttonHover = { scale: 1.02, transition: { duration: 0.2 } };

  return (
    <div
      className="min-h-screen bg-cyber-grid flex flex-col items-center justify-center p-6 relative"
      style={{ backgroundColor: 'var(--bg-primary)' }}
    >

      {/* Floating ambient particles */}
      <FloatingParticles />

      {/* Red vignette overlay when attack is detected */}
      <AnimatePresence>
        {!isSafe && appState === "active" && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="vignette-danger"
          />
        )}
      </AnimatePresence>

      {/* ============================================================
          BOOT-UP SEQUENCE — Each section has its own explicit
          delay-based animation props so they cascade independently.
          Header (0.1s) → Video Panel (0.4s) → Side Panel (0.7s)
          ============================================================ */}

      {/* HEADER */}
      <motion.div {...headerAnim} className="w-full max-w-6xl flex justify-between items-end mb-8 z-10 relative">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <Terminal size={28} className="text-blue-500" />
            <h1 className="text-4xl font-black tracking-tight uppercase drop-shadow-[0_0_10px_rgba(59,130,246,0.5)]" style={{ color: 'var(--text-primary)' }}>
              VSA <span className="text-blue-500">Sentinel</span>
            </h1>
          </div>
          <p className="text-sm font-mono tracking-widest uppercase ml-10" style={{ color: 'var(--text-muted)' }}>Continuous Multimodal Authentication</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Dark/Light mode toggle button */}
          <motion.button
            whileHover={buttonHover} whileTap={buttonTap}
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="p-2.5 rounded-md border font-mono text-xs uppercase tracking-wider transition-colors"
            style={{
              backgroundColor: 'var(--glass-bg)',
              borderColor: 'var(--glass-border)',
              color: 'var(--text-secondary)'
            }}
            title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </motion.button>
          {/* Connection indicator with breathing glow */}
          <div className={`px-4 py-2 rounded-md font-mono text-xs border uppercase tracking-wider ${isConnected ? "bg-green-950/50 text-green-400 border-green-500/50 shadow-[0_0_10px_rgba(16,185,129,0.3)] animate-breathe" : "bg-red-950/50 text-red-400 border-red-500/50 shadow-[0_0_10px_rgba(225,29,72,0.3)]"}`}>
            {isConnected ? "● SYSTEM ONLINE" : "■ DISCONNECTED"}
          </div>
        </div>
      </motion.div>

      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-8 z-10 relative">

        {/* MAIN HUD VIDEO FEED — with boot-up animation */}
        <motion.div {...videoPanelAnim} className="lg:col-span-2 relative glass-panel rounded-lg overflow-hidden aspect-video flex items-center justify-center p-1">

          {/* HUD corners with breathing glow */}
          <div className={`hud-corner top-4 left-4 border-t-2 border-l-2 animate-breathe ${!isSafe ? 'border-red-500' : ''}`}></div>
          <div className={`hud-corner top-4 right-4 border-t-2 border-r-2 animate-breathe ${!isSafe ? 'border-red-500' : ''}`}></div>
          <div className={`hud-corner bottom-4 left-4 border-b-2 border-l-2 animate-breathe ${!isSafe ? 'border-red-500' : ''}`}></div>
          <div className={`hud-corner bottom-4 right-4 border-b-2 border-r-2 animate-breathe ${!isSafe ? 'border-red-500' : ''}`}></div>

          {/* Scanner line */}
          {(appState === "uncalibrated" || appState === "calibrating") && <div className="scan-line"></div>}

          <video ref={videoRef} autoPlay playsInline muted style={{ width: "1px", height: "1px", opacity: 0, position: "absolute" }} />
          <canvas ref={canvasRef} width="640" height="480" className="hidden" />

          {displayImage ? (
            <img src={displayImage} alt="Live Stream" className="w-full h-full object-cover scale-x-[-1] rounded" />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-blue-500/50 font-mono animate-pulse">
              <ScanFace size={48} className="mb-4" />
              WAITING FOR VIDEO STREAM...
            </div>
          )}

          {/* Face alignment overlay — CSS oval outlining a human head shape */}
          {appState !== "active" && appState !== "calibration_complete" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-20 bg-slate-900/10">
              <div className="face-oval"></div>
              <div className="mt-6 px-6 py-2 rounded text-xs font-mono font-bold tracking-widest uppercase shadow-[0_0_10px_rgba(59,130,246,0.2)]"
                style={{ backgroundColor: 'var(--btn-secondary-bg)', border: '1px solid var(--btn-secondary-border)', color: 'var(--hud-color)' }}
              >
                {appState === "calibrating" ? "MAPPING BIOMETRIC PROFILE..." : "Align face in oval to commence zero-shot profiling"}
              </div>
            </div>
          )}

          {/* Attack alert with glitch effect */}
          <AnimatePresence>
            {!isSafe && appState === "active" && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 border-4 border-red-500 bg-red-900/20 z-40 pointer-events-none flex items-center justify-center"
              >
                <div className={`bg-red-600/90 backdrop-blur-md border border-red-400 text-white px-8 py-4 rounded font-mono font-bold flex items-center justify-center gap-3 shadow-[0_0_30px_rgba(239,68,68,0.8)] text-xl tracking-wider ${glitchActive ? 'glitch-text' : ''}`}>
                  <ShieldAlert size={32} />
                  {verdict === "FACE OBSCURED" ? "FACE OBSCURED - SECURING STREAM" : "BIOMETRIC INCOHERENCE DETECTED"}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* METRICS SIDE PANEL — with boot-up animation */}
        <motion.div {...sidePanelAnim} className="flex flex-col gap-6">

          {appState !== "active" ? (
            /* Wrapped in Tilt for 3D hover effect */
            <Tilt tiltMaxAngleX={5} tiltMaxAngleY={8} glareEnable={true} glareMaxOpacity={0.08} glareColor="#38BDF8" glarePosition="all" scale={1.02}>
              <div className="glass-panel-premium p-8 rounded-lg flex flex-col items-center justify-center text-center h-full">

                {appState === "uncalibrated" && (
                  <>
                    <ScanFace size={72} className="text-blue-500 mb-6 drop-shadow-[0_0_15px_rgba(59,130,246,0.6)]" />
                    <h2 className="text-2xl font-black mb-2 uppercase tracking-wide" style={{ color: 'var(--text-primary)' }}>System Standby</h2>
                    <p className="text-sm mb-10 font-mono" style={{ color: 'var(--text-secondary)' }}>Awaiting primary biometric mapping.</p>
                    <motion.button
                      whileHover={buttonHover} whileTap={buttonTap}
                      onClick={startCalibration}
                      className="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white font-mono font-bold rounded shadow-[0_0_20px_rgba(59,130,246,0.4)] transition-colors"
                    >
                      INITIALIZE CALIBRATION
                    </motion.button>
                  </>
                )}

                {appState === "calibrating" && (
                  <div className="w-full flex flex-col items-center">
                    <div className="text-7xl font-black text-blue-500 mb-2 drop-shadow-[0_0_15px_rgba(59,130,246,0.6)]">{secondsRemaining}s</div>
                    <h2 className="text-lg font-bold mb-1 uppercase tracking-wide" style={{ color: 'var(--text-primary)' }}>Speak Naturally</h2>
                    <p className="text-xs mb-10 font-mono" style={{ color: 'var(--text-secondary)' }}>Adapting MAML neural weights...</p>
                    <div className="w-full">
                      <div className="flex justify-between text-[10px] text-blue-400 mb-2 font-mono uppercase tracking-widest">
                        <span>Extracting Optical Flow</span>
                        <span>{calibProgress}%</span>
                      </div>
                      <div className="h-1 rounded-full overflow-hidden relative" style={{ backgroundColor: 'var(--bar-bg)' }}>
                        <motion.div className="absolute top-0 left-0 h-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.8)]" initial={{ width: 0 }} animate={{ width: `${calibProgress}%` }} />
                      </div>
                    </div>
                  </div>
                )}

                {/* Calibration complete — spring bounce "lock-in" animation */}
                {appState === "calibration_complete" && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 200, damping: 15 }}
                    className="flex flex-col items-center"
                  >
                    <CheckCircle2 size={72} className="text-emerald-500 mb-6 drop-shadow-[0_0_15px_rgba(16,185,129,0.6)]" />
                    <h2 className="text-xl font-bold mb-2 uppercase tracking-wide" style={{ color: 'var(--text-primary)' }}>Profile Secured</h2>
                    <p className="text-sm mb-10 font-mono" style={{ color: 'var(--text-secondary)' }}>Base weights locked. Monitoring active.</p>
                    <motion.button
                      whileHover={buttonHover} whileTap={buttonTap}
                      onClick={() => setAppState("active")}
                      className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold rounded shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-colors"
                    >
                      ENGAGE LIVE DEFENSE
                    </motion.button>
                  </motion.div>
                )}

              </div>
            </Tilt>
          ) : (
            <>
              {/* TRUST SCORE GAUGE — Wrapped in Tilt */}
              <Tilt tiltMaxAngleX={4} tiltMaxAngleY={6} glareEnable={true} glareMaxOpacity={0.06} glareColor={themeColor} glarePosition="all">
                <motion.div
                  className="glass-panel-premium p-6 rounded-lg flex flex-col items-center relative overflow-hidden"
                  /* Shake effect when trust score drops below 50 */
                  animate={trustScore < 50 ? { x: [0, -3, 3, -2, 2, 0] } : {}}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                >
                  <h2 className="text-xs font-mono mb-6 uppercase tracking-widest flex items-center gap-2" style={{ color: 'var(--text-muted)' }}><Activity size={14} /> Confidence Metric</h2>
                  <div className="w-48 h-48 relative drop-shadow-[0_0_15px_rgba(16,185,129,0.3)]">
                    <CircularProgressbar
                      value={trustScore}
                      text={`${trustScore}%`}
                      styles={buildStyles({
                        pathColor: themeColor,
                        textColor: 'var(--progress-text)',
                        trailColor: 'var(--progress-trail)',
                        pathTransitionDuration: 0.5,
                        textSize: '22px'
                      })}
                    />
                  </div>
                  {/* Verdict badge with glitch effect */}
                  <motion.div
                    animate={{ scale: isSafe ? 1 : 1.05 }}
                    transition={{ type: "spring", stiffness: 300, damping: 15 }}
                    className={`mt-8 px-8 py-2 rounded border font-mono font-bold text-lg flex items-center gap-3 uppercase tracking-wider ${glitchActive ? 'glitch-text' : ''} ${isSafe ? "bg-emerald-950/50 text-emerald-400 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.2)]" : "bg-red-950/50 text-red-500 border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.4)]"}`}
                  >
                    {isSafe ? <ShieldCheck size={22} /> : <ShieldAlert size={22} />} {verdict}
                  </motion.div>
                </motion.div>
              </Tilt>

              {/* STREAM ANALYSIS — ECG-style area chart */}
              <Tilt tiltMaxAngleX={3} tiltMaxAngleY={5} glareEnable={true} glareMaxOpacity={0.04} glareColor="#38BDF8" glarePosition="all">
                <div className="glass-panel-premium p-6 rounded-lg">
                  <h3 className="text-xs font-mono uppercase tracking-widest mb-4 pb-2" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--glass-border)' }}>Component Analysis</h3>

                  {/* Real-time anomaly chart */}
                  <div className="w-full h-24 mb-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="lipGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#e11d48" stopOpacity={0.4} />
                            <stop offset="95%" stopColor="#e11d48" stopOpacity={0} />
                          </linearGradient>
                          <linearGradient id="eyeGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#38BDF8" stopOpacity={0.4} />
                            <stop offset="95%" stopColor="#38BDF8" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <YAxis domain={[0, 1]} hide />
                        <Area type="monotone" dataKey="lip" stroke="#e11d48" fill="url(#lipGrad)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Area type="monotone" dataKey="eye" stroke="#38BDF8" fill="url(#eyeGrad)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Chart legend */}
                  <div className="flex justify-between text-[10px] font-mono uppercase tracking-widest mb-4">
                    <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-rose-500 inline-block"></span> <span style={{ color: 'var(--text-muted)' }}>LIP_STREAM</span></span>
                    <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-sky-400 inline-block"></span> <span style={{ color: 'var(--text-muted)' }}>EYE_STREAM</span></span>
                  </div>

                  {/* Anomaly bars */}
                  <div className="mb-4">
                    <div className="flex justify-between font-mono text-xs mb-2">
                      <span style={{ color: 'var(--text-secondary)' }}>LIP_ANOMALY</span>
                      <span className={lipProb > 0.75 ? "text-red-400" : "text-blue-400"}>{(lipProb * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bar-bg)' }}>
                      <motion.div className={`h-full ${lipProb > 0.75 ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]" : "bg-blue-500"}`} initial={{ width: 0 }} animate={{ width: `${lipProb * 100}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between font-mono text-xs mb-2">
                      <span style={{ color: 'var(--text-secondary)' }}>EYE_ANOMALY</span>
                      <span className={eyeProb > 0.75 ? "text-red-400" : "text-blue-400"}>{(eyeProb * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bar-bg)' }}>
                      <motion.div className={`h-full ${eyeProb > 0.75 ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]" : "bg-blue-500"}`} initial={{ width: 0 }} animate={{ width: `${eyeProb * 100}%` }} />
                    </div>
                  </div>
                </div>
              </Tilt>

              {/* ATTACK TOGGLE */}
              <motion.button
                whileHover={buttonHover} whileTap={buttonTap}
                onClick={toggleInjection}
                className={`w-full py-5 rounded uppercase font-mono font-bold flex items-center justify-center gap-3 transition-colors border ${isInjecting ? "bg-slate-800 text-slate-400 border-slate-600 hover:bg-slate-700" : "bg-red-950/80 text-red-500 border-red-500/50 hover:bg-red-900/80 shadow-[0_0_15px_rgba(239,68,68,0.3)] hover:shadow-[0_0_25px_rgba(239,68,68,0.5)]"}`}
              >
                <Zap size={22} fill={isInjecting ? "none" : "currentColor"} className={!isInjecting ? "animate-pulse" : ""} />
                {isInjecting ? "ABORT HIJACK SIMULATION" : "EXECUTE SESSION HIJACK"}
              </motion.button>
            </>
          )}
        </motion.div>
      </div>
    </div>
  );
}

export default App;