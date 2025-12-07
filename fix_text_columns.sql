-- Fix TEXT columns that should be DOUBLE PRECISION
DO $$ 
DECLARE 
    col_name text;
BEGIN
    FOR col_name IN 
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema='feature_store' 
          AND table_name='features' 
          AND data_type='text'
    LOOP
        EXECUTE format(
            'ALTER TABLE feature_store.features ALTER COLUMN %I TYPE DOUBLE PRECISION USING NULLIF(%I, '''')::DOUBLE PRECISION',
            col_name, col_name
        );
        RAISE NOTICE 'Converted column % to DOUBLE PRECISION', col_name;
    END LOOP;
END $$;
