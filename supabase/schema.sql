create table if not exists public.analyses (
  run_id text primary key,
  owner_user_id text not null,
  analysis_version text,
  score integer,
  shooting_side text,
  camera_view text not null default 'side',
  report jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists analyses_owner_created_at_idx
  on public.analyses (owner_user_id, created_at desc);

alter table public.analyses enable row level security;

drop policy if exists "Users can read their own analyses" on public.analyses;
create policy "Users can read their own analyses"
  on public.analyses
  for select
  using (auth.uid()::text = owner_user_id);

drop policy if exists "Users can delete their own analyses" on public.analyses;
create policy "Users can delete their own analyses"
  on public.analyses
  for delete
  using (auth.uid()::text = owner_user_id);

insert into storage.buckets (id, name, public)
values ('shot-analyses', 'shot-analyses', false)
on conflict (id) do update set public = excluded.public;
