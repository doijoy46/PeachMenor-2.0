'use client';

import { useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import { signInWithPopup, signOut } from 'firebase/auth';
import { auth, googleProvider } from '@/lib/firebase';
import { useAuth } from './AuthProvider';

function DoorPanel({ side }: { side: 'left' | 'right' }) {
  const isLeft = side === 'left';

  return (
    <div
      className="relative h-full w-full select-none overflow-hidden"
      style={{
        background: `linear-gradient(
          ${isLeft ? '108deg' : '72deg'},
          #2c1505 0%,
          #5c2d0e 12%,
          #3d1f08 25%,
          #6b3515 40%,
          #3a1c07 55%,
          #5a2a0c 70%,
          #3d1f08 85%,
          #2c1505 100%
        )`,
      }}
    >
      {/* Wood grain overlay */}
      <div
        className="absolute inset-0 opacity-[0.15]"
        style={{
          backgroundImage: `repeating-linear-gradient(
            ${isLeft ? '94deg' : '86deg'},
            transparent,
            transparent 3px,
            rgba(0,0,0,0.35) 3px,
            rgba(0,0,0,0.35) 4px
          )`,
        }}
      />

      {/* Outer frame bevel */}
      <div
        className="absolute"
        style={{
          inset: '14px',
          boxShadow: `
            inset 2px 2px 4px rgba(255,210,120,0.1),
            inset -2px -2px 4px rgba(0,0,0,0.5),
            0 0 0 1px rgba(0,0,0,0.3)
          `,
        }}
      />

      {/* Upper raised panel */}
      <RaisedPanel top="10%" left="16%" right="16%" height="33%" />

      {/* Lower raised panel */}
      <RaisedPanel bottom="10%" left="16%" right="16%" height="33%" />

      {/* Brass door knob */}
      <div
        className="absolute top-1/2 -translate-y-1/2"
        style={{ [isLeft ? 'right' : 'left']: '7%' }}
      >
        <div
          className="h-6 w-6 rounded-full"
          style={{
            background:
              'radial-gradient(circle at 35% 35%, #f5d68a, #c9920d 55%, #8b6508)',
            boxShadow:
              '0 2px 8px rgba(0,0,0,0.7), inset 0 1px 2px rgba(255,255,255,0.25)',
          }}
        />
      </div>

      {/* Inner-edge depth shadow */}
      <div
        className="absolute inset-y-0 w-16"
        style={{
          [isLeft ? 'right' : 'left']: 0,
          background: `linear-gradient(${isLeft ? 'to left' : 'to right'}, rgba(0,0,0,0.55), transparent)`,
        }}
      />

      {/* Top/bottom edge darkening */}
      <div
        className="absolute inset-x-0 top-0 h-8"
        style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.4), transparent)' }}
      />
      <div
        className="absolute inset-x-0 bottom-0 h-8"
        style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.4), transparent)' }}
      />
    </div>
  );
}

function RaisedPanel({
  top,
  bottom,
  left,
  right,
  height,
}: {
  top?: string;
  bottom?: string;
  left: string;
  right: string;
  height: string;
}) {
  return (
    <div className="absolute" style={{ top, bottom, left, right, height }}>
      <div
        className="h-full w-full"
        style={{
          background: 'rgba(0,0,0,0.2)',
          boxShadow: `
            inset 1px 1px 0 rgba(255,210,120,0.12),
            inset -1px -1px 0 rgba(0,0,0,0.45),
            0 0 0 1px rgba(0,0,0,0.25)
          `,
        }}
      />
    </div>
  );
}

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
      <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
      <path d="M3.964 10.707A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.707V4.961H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.039l3.007-2.332z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.961L3.964 7.293C4.672 5.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
    </svg>
  );
}

function AuthButton() {
  const { user } = useAuth();

  if (user) {
    return (
      <div className="flex items-center gap-3">
        {user.photoURL && (
          <Image
            src={user.photoURL}
            alt={user.displayName ?? 'User'}
            width={32}
            height={32}
            className="rounded-full"
          />
        )}
        <button
          className="text-sm tracking-widest uppercase transition-colors duration-200 hover:opacity-70"
          style={{ color: '#3d1f08' }}
          onClick={() => signOut(auth)}
        >
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <button
      className="flex items-center gap-2 rounded-full border bg-white px-5 py-2 text-sm font-medium tracking-wide shadow-sm transition-shadow duration-200 hover:shadow-md"
      style={{ borderColor: '#dadce0', color: '#3c4043' }}
      onClick={() => signInWithPopup(auth, googleProvider)}
    >
      <GoogleIcon />
      Sign in with Google
    </button>
  );
}

function BrandingSection({ ctaVisible }: { ctaVisible: boolean }) {
  return (
    <div className="mt-auto mb-auto flex flex-col items-center gap-4 px-8 text-center">
      <h1
        className="text-6xl font-light tracking-widest"
        style={{ color: '#3d1f08', fontFamily: 'Georgia, serif' }}
      >
        PeachMenor
      </h1>
      <p
        className="text-lg tracking-[0.25em] uppercase"
        style={{ color: '#8b6508' }}
      >
        Your Digital Wardrobe
      </p>

      <div
        className="mt-4 flex flex-col items-center gap-4 transition-opacity duration-700"
        style={{ opacity: ctaVisible ? 1 : 0, pointerEvents: ctaVisible ? 'auto' : 'none' }}
      >
        <AuthButton />

        <label
          className="cursor-pointer rounded-full border-2 px-10 py-3 text-sm tracking-widest uppercase transition-colors duration-200 hover:bg-amber-900 hover:text-amber-50"
          style={{ borderColor: '#3d1f08', color: '#3d1f08' }}
        >
          Upload to Catalog
          <input
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={e => {
              const files = Array.from(e.target.files ?? []);
              console.log('Selected files:', files.map(f => f.name));
              e.target.value = '';
            }}
          />
        </label>
      </div>
    </div>
  );
}

function WardrobeInterior({ ctaVisible }: { ctaVisible: boolean }) {
  return (
    <div
      className="absolute inset-0 flex flex-col items-center"
      style={{
        background: '#f5efe6',
        backgroundImage: `repeating-linear-gradient(
          90deg,
          transparent,
          transparent 60px,
          rgba(180,140,100,0.08) 60px,
          rgba(180,140,100,0.08) 61px
        )`,
      }}
    >
      {/* Top wood bar */}
      <div
        className="w-full"
        style={{
          height: '28px',
          background: 'linear-gradient(to bottom, #5c2d0e, #3d1f08)',
          boxShadow: '0 4px 12px rgba(0,0,0,0.35)',
        }}
      />

      {/* Clothes rod */}
      <div className="relative w-full" style={{ height: '80px' }}>
        <div
          className="absolute left-0 right-0"
          style={{
            top: '50%',
            height: '6px',
            background: 'linear-gradient(to bottom, #c9920d, #8b6508)',
            boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
          }}
        />
        {/* Clothes hangers */}
        {[12, 25, 40, 55, 68, 80].map((pct, i) => (
          <ClothingItem key={i} left={`${pct}%`} index={i} />
        ))}
      </div>

      {/* Branding */}
      <BrandingSection ctaVisible={ctaVisible} />
    </div>
  );
}

function ClothingItem({ left, index }: { left: string; index: number }) {
  const colors = ['#c8a882', '#b08860', '#d4b896', '#a07840', '#c0926a', '#b89060'];
  const heights = [90, 110, 80, 100, 95, 85];
  const widths = [44, 50, 38, 48, 42, 46];

  return (
    <div className="absolute flex flex-col items-center" style={{ left, top: '50%' }}>
      {/* Hook */}
      <div
        style={{
          width: '1px',
          height: '16px',
          background: '#8b6508',
          transform: 'translateX(-50%)',
        }}
      />
      {/* Garment silhouette */}
      <div
        style={{
          width: `${widths[index]}px`,
          height: `${heights[index]}px`,
          background: colors[index],
          borderRadius: '4px 4px 6px 6px',
          opacity: 0.75,
          transform: 'translateX(-50%)',
          boxShadow: '2px 2px 6px rgba(0,0,0,0.15)',
        }}
      />
    </div>
  );
}

export default function WardrobeHero() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [angle, setAngle] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (rafRef.current !== null) return;
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        const el = containerRef.current;
        if (!el) return;
        const { top, height } = el.getBoundingClientRect();
        const scrollRange = height - window.innerHeight;
        const progress = Math.max(0, Math.min(1, -top / scrollRange));
        setAngle(progress * 115);
      });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const ctaVisible = angle > 95;

  return (
    <div ref={containerRef} style={{ height: '250vh' }}>
      <div className="sticky top-0 h-screen overflow-hidden">
        {/* Wardrobe interior — behind the doors */}
        <WardrobeInterior ctaVisible={ctaVisible} />

        {/* Door container with shared perspective */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{ perspective: '1400px', perspectiveOrigin: '50% 50%' }}
        >
          {/* Center seam */}
          <div
            className="absolute inset-y-0 left-1/2 z-20 w-px -translate-x-px bg-black/60"
            style={{ opacity: Math.max(0, 1 - angle / 30) }}
          />

          {/* Left door */}
          <div
            className="absolute left-0 top-0 h-full w-1/2"
            style={{
              transformOrigin: 'left center',
              transform: `rotateY(${-angle}deg)`,
              zIndex: 10,
            }}
          >
            <DoorPanel side="left" />
          </div>

          {/* Right door */}
          <div
            className="absolute right-0 top-0 h-full w-1/2"
            style={{
              transformOrigin: 'right center',
              transform: `rotateY(${angle}deg)`,
              zIndex: 10,
            }}
          >
            <DoorPanel side="right" />
          </div>
        </div>
      </div>
    </div>
  );
}
