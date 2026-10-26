// A spring grid: points repel the pointer, then settle back into place.
(()=>{
    const canvas=document.getElementById('grid');if(!canvas?.getContext)return;
    const ctx=canvas.getContext('2d');if(!ctx)return;
    const motion=window.matchMedia('(prefers-reduced-motion: reduce)'),fine=window.matchMedia('(pointer: fine)');
    let width=0,height=0,points=[],frame=0,lastTime=0;
    const pointer={x:-1000,y:-1000,active:false};
    function draw(){
        ctx.clearRect(0,0,width,height);
        for(const p of points){
            const displacement=Math.hypot(p.x-p.ox,p.y-p.oy);
            ctx.fillStyle=`rgba(255,255,255,${Math.min(.43,.16+displacement*.0035)})`;
            ctx.beginPath();ctx.arc(p.x,p.y,displacement>1?1.15:1,0,Math.PI*2);ctx.fill();
        }
    }
    function request(){if(!frame&&!document.hidden)frame=requestAnimationFrame(tick)}
    function tick(time){
        frame=0;if(document.hidden)return;
        const dt=lastTime?Math.min(2,(time-lastTime)/16.667):1;lastTime=time;let moving=false;
        for(const p of points){
            let tx=p.ox,ty=p.oy;
            if(pointer.active&&!motion.matches){
                const dx=p.ox-pointer.x,dy=p.oy-pointer.y,d=Math.hypot(dx,dy),radius=155;
                if(d<radius){const force=(1-d/radius)**2*67;const angle=d>.1?Math.atan2(dy,dx):(p.ox+p.oy)%6.28;tx+=Math.cos(angle)*force;ty+=Math.sin(angle)*force}
            }
            p.vx+=(tx-p.x)*.055*dt;p.vy+=(ty-p.y)*.055*dt;p.vx*=Math.pow(.73,dt);p.vy*=Math.pow(.73,dt);p.x+=p.vx*dt;p.y+=p.vy*dt;
            if(Math.abs(tx-p.x)+Math.abs(ty-p.y)+Math.abs(p.vx)+Math.abs(p.vy)>.06)moving=true;
            else{p.x=tx;p.y=ty;p.vx=p.vy=0}
        }
        draw();if(moving&&!motion.matches)request();else lastTime=0;
    }
    function resize(){
        width=window.innerWidth;height=window.innerHeight;const dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
        points=[];const gap=30;for(let y=15;y<height;y+=gap)for(let x=15;x<width;x+=gap)points.push({ox:x,oy:y,x,y,vx:0,vy:0});draw();request();
    }
    window.addEventListener('pointermove',e=>{if(!fine.matches||motion.matches||e.pointerType==='touch')return;pointer.x=e.clientX;pointer.y=e.clientY;pointer.active=true;request()},{passive:true});
    function leave(){pointer.active=false;request()}
    document.documentElement.addEventListener('pointerleave',leave);window.addEventListener('blur',leave);window.addEventListener('resize',resize,{passive:true});
    document.addEventListener('visibilitychange',()=>{if(document.hidden){if(frame)cancelAnimationFrame(frame);frame=0;pointer.active=false;lastTime=0}else request()});
    motion.addEventListener('change',()=>{pointer.active=false;resize()});resize();
})();
