-- ════════════════════════════════════════════════════════════════
--  Neural Arena — Schéma Supabase pour le stockage des cerveaux
--  À exécuter dans : Supabase Dashboard → SQL Editor → New query
-- ════════════════════════════════════════════════════════════════

-- 1. Table des cerveaux ------------------------------------------------
create table if not exists public.brains (
  id          text primary key,            -- id local du cerveau (brain_..., cloud_...)
  user_id     text not null,               -- propriétaire (auth.uid() ou id anonyme local)
  name        text,
  creator     text,
  elo         real        default 1200,
  fitness     real        default 0,
  generation  int         default 1,
  params      int         default 0,
  personality text,
  topology    text,
  data        jsonb       not null,         -- brain.serialize()
  updated_at  timestamptz default now()
);

create index if not exists brains_user_idx on public.brains (user_id);
create index if not exists brains_elo_idx  on public.brains (elo desc);

-- 2. Row Level Security ------------------------------------------------
alter table public.brains enable row level security;

-- Lecture publique : tout le monde voit la galerie communautaire.
drop policy if exists "brains_public_read" on public.brains;
create policy "brains_public_read"
  on public.brains for select
  using (true);

-- ────────────────────────────────────────────────────────────────────
--  MODE A — LOCAL / ANONYME (par défaut dans l'app)
--  L'app envoie un user_id anonyme généré sur l'appareil.
--  Écriture ouverte : simple à mettre en place, suffisant pour un proto.
--  ⚠ N'importe qui avec la clé anon peut écrire. Pour un vrai
--    cloisonnement, utilise le MODE B ci-dessous.
-- ────────────────────────────────────────────────────────────────────
drop policy if exists "brains_anon_write"  on public.brains;
create policy "brains_anon_write"
  on public.brains for insert
  with check (true);

drop policy if exists "brains_anon_update" on public.brains;
create policy "brains_anon_update"
  on public.brains for update
  using (true);

drop policy if exists "brains_anon_delete" on public.brains;
create policy "brains_anon_delete"
  on public.brains for delete
  using (true);

-- ────────────────────────────────────────────────────────────────────
--  MODE B — AUTH (recommandé en production)
--  1) Active "Anonymous sign-ins" :
--       Dashboard → Authentication → Providers → Anonymous → Enable
--  2) Dans l'app : onglet ☁ Cloud → "🔐 Compte anonyme"
--  3) Remplace les policies du MODE A par celles-ci (décommente) :
--
-- drop policy if exists "brains_anon_write"  on public.brains;
-- drop policy if exists "brains_anon_update" on public.brains;
-- drop policy if exists "brains_anon_delete" on public.brains;
--
-- create policy "brains_owner_write"
--   on public.brains for insert
--   with check (auth.uid()::text = user_id);
--
-- create policy "brains_owner_update"
--   on public.brains for update
--   using (auth.uid()::text = user_id)
--   with check (auth.uid()::text = user_id);
--
-- create policy "brains_owner_delete"
--   on public.brains for delete
--   using (auth.uid()::text = user_id);
-- ────────────────────────────────────────────────────────────────────
