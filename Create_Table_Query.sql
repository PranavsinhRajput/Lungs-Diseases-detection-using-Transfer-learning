-- Table: public.predictions

-- DROP TABLE IF EXISTS public.predictions;

CREATE TABLE IF NOT EXISTS public.predictions
(
    id integer NOT NULL DEFAULT nextval('predictions_id_seq'::regclass),
    name character varying(100) COLLATE pg_catalog."default" NOT NULL,
    age integer NOT NULL,
    gender character varying(20) COLLATE pg_catalog."default" NOT NULL,
    prediction_result character varying(20) COLLATE pg_catalog."default" NOT NULL,
    disease_name character varying(50) COLLATE pg_catalog."default" NOT NULL,
    confidence character varying(20) COLLATE pg_catalog."default" NOT NULL,
    image_path character varying(255) COLLATE pg_catalog."default" NOT NULL,
    prediction_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT predictions_pkey PRIMARY KEY (id)
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.predictions
    OWNER to postgres;
