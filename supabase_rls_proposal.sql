-- ============================================================================
-- PROPOSAL — NOT YET RUN. Review before pasting into the Supabase SQL editor.
-- Row-level security (RLS) for the FlinMind tables.
-- ============================================================================
--
-- WHY
--   A read-only check on 2026-11-22 showed the key the backend uses (role claim
--   "anon") can SELECT every row of `documents` (including chunk text and
--   embeddings) and `document_registry`. The same key was also used to INSERT
--   and DELETE rows during a re-index. Supabase treats the anon key as public,
--   so anyone holding the project URL + anon key can currently read, change or
--   wipe all documents.
--
-- THE PLAN
--   1. The backend switches to the service_role key (a server-only secret).
--      service_role bypasses RLS automatically, so the API keeps working.
--   2. RLS is enabled on both tables with NO policies for anon/authenticated,
--      which means "deny everything" for those roles.
--   3. Direct table privileges and the match_documents function are revoked from
--      anon/authenticated as a second layer.
--
-- !! ORDER MATTERS !!
--   Do step 1 FIRST (local .env and the Render environment variables):
--       SUPABASE_KEY=<service_role key from Supabase → Project Settings → API>
--   If you enable RLS while the backend still uses the anon key, every query
--   returns no rows and uploads fail. Then redeploy and confirm the API works.
--
-- TRADEOFFS
--   + Data is no longer readable/writable by anyone who has the anon key.
--   + No policy logic to get wrong: only the server (service_role) touches data.
--   - The service_role key bypasses ALL RLS. It must never reach the browser,
--     a public repo, or logs. Treat it like a database password.
--   - Tenant isolation is still enforced only by the API's X-Tenant-ID header,
--     not by the database. Real per-tenant isolation needs user authentication
--     (e.g. Supabase Auth) plus policies keyed on the user's tenant — see the
--     commented example at the end. That is a larger change.
--   - If match_documents is SECURITY DEFINER it ignores RLS for its caller, so
--     revoking EXECUTE from anon/authenticated (below) is what protects it.
--
-- Check current state first with the read-only queries in the report
-- (pg_tables.rowsecurity, pg_policies, pg_get_functiondef). If pg_policies
-- lists permissive policies for anon/authenticated on these tables, drop them:
--     DROP POLICY "<policy name>" ON public.documents;
-- ============================================================================

BEGIN;

-- 2. Enable RLS (no policies = deny for anon/authenticated; service_role bypasses)
ALTER TABLE public.documents         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_registry ENABLE ROW LEVEL SECURITY;

-- 3a. Belt and braces: remove direct table privileges from the public roles
REVOKE ALL ON TABLE public.documents         FROM anon, authenticated;
REVOKE ALL ON TABLE public.document_registry FROM anon, authenticated;

-- 3b. Revoke the vector-search function from public roles (every overload)
DO $$
DECLARE fn regprocedure;
BEGIN
  FOR fn IN
    SELECT p.oid::regprocedure
    FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public' AND p.proname = 'match_documents'
  LOOP
    EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC, anon, authenticated', fn);
    EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', fn);
  END LOOP;
END $$;

COMMIT;

-- ---------------------------------------------------------------------------
-- Verify afterwards (expect rowsecurity = true, no anon/authenticated policies)
-- ---------------------------------------------------------------------------
-- SELECT tablename, rowsecurity FROM pg_tables
--  WHERE schemaname = 'public' AND tablename IN ('documents', 'document_registry');
-- SELECT tablename, policyname, roles, cmd FROM pg_policies WHERE schemaname = 'public';

-- ---------------------------------------------------------------------------
-- ROLLBACK (only if something breaks and you need the old behaviour back)
-- ---------------------------------------------------------------------------
-- ALTER TABLE public.documents         DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.document_registry DISABLE ROW LEVEL SECURITY;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON public.documents, public.document_registry TO anon;
-- (and re-grant EXECUTE on match_documents to anon)

-- ---------------------------------------------------------------------------
-- FUTURE (not part of this proposal): per-tenant access with Supabase Auth.
-- Requires users to log in and a tenant_id claim in their JWT.
-- ---------------------------------------------------------------------------
-- CREATE POLICY tenant_read ON public.documents FOR SELECT TO authenticated
--   USING (tenant_id = (auth.jwt() ->> 'tenant_id'));
-- CREATE POLICY tenant_read ON public.document_registry FOR SELECT TO authenticated
--   USING (tenant_id = (auth.jwt() ->> 'tenant_id'));
