---
title: Pgweather
emoji: 🏆
colorFrom: gray
colorTo: yellow
sdk: streamlit
sdk_version: 1.34.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Fly.io + Supabase deployment

This app now reads the database connection from `DATABASE_URL`.

1. Create/update a Fly app config (`fly.toml` is included).
2. Set the Supabase Postgres connection string as a Fly secret:

```bash
fly secrets set DATABASE_URL="postgresql://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres?sslmode=require" -a pgweather
```

3. Deploy:

```bash
fly deploy -a pgweather
```

4. Verify:

```bash
fly logs -a pgweather
```
