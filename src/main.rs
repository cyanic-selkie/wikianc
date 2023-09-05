#![feature(iter_array_chunks)]

use apache_avro::{from_value, Reader};
use arrow2::{
    array::Array,
    chunk::Chunk,
    datatypes::*,
    io::parquet::write::{
        transverse, CompressionOptions, Encoding, FileWriter, RowGroupIterator, Version,
        WriteOptions,
    },
};
use arrow2_convert::{
    serialize::{FlattenChunk, TryIntoArrow},
    ArrowDeserialize, ArrowField, ArrowSerialize,
};
use clap::Parser;
use hashbrown::HashMap;
use htmlescape::decode_html;
use itertools::Itertools;
use lazy_regex::regex_replace_all;
use rand::distributions::{Bernoulli, Distribution};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the mappings between Wikipedia's titles and Wikidata's QIDs in the Apache Avro format.
    #[arg(long)]
    input_wiki2qid: String,
    /// Path to the Wikipedia dump in the ndjson format.
    #[arg(long)]
    input_wiki: String,
    /// Path to the output directory.
    #[arg(long)]
    output_dir: String,
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn normalized_deserializer<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let string: String = Deserialize::deserialize(deserializer)?;
    let normalized = string.nfc().collect::<String>();
    let decoded = decode_html(&normalized);
    let string = match decoded {
        Err(_) => normalized,
        Ok(x) => x,
    };
    let string = regex_replace_all!(r#"(<ref .+>|</ref>)"#i, &string, |_, _| "");
    let string = regex_replace_all!(r#"(<!--.+-->)"#i, &string, |_, _| "");
    Ok(string.trim().to_owned())
}

fn normalized_deserializer_optional<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let string: String = Deserialize::deserialize(deserializer)?;
    let normalized = string.nfc().collect::<String>();
    let decoded = decode_html(&normalized);
    let string = match decoded {
        Err(_) => normalized,
        Ok(x) => x,
    };
    let string = regex_replace_all!(r#"(<ref>.*</ref>)"#i, &string, |_, _| "");
    let string = regex_replace_all!(r#"(<!--.+-->)"#i, &string, |_, _| "");
    Ok(Some(string.trim().to_owned()))
}

#[derive(Serialize, Deserialize, Debug)]
struct Link {
    #[serde(default)]
    #[serde(deserialize_with = "normalized_deserializer_optional")]
    text: Option<String>,
    #[serde(deserialize_with = "normalized_deserializer")]
    title: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Sentence {
    #[serde(deserialize_with = "normalized_deserializer")]
    text: String,
    #[serde(rename(deserialize = "links"))]
    anchors: Vec<Link>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Section {
    #[serde(deserialize_with = "normalized_deserializer")]
    title: String,
    depth: u32,
    paragraphs: Vec<Vec<Sentence>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Article {
    #[serde(deserialize_with = "normalized_deserializer")]
    title: String,
    #[serde(rename = "pageID")]
    pageid: u32,
    sections: Vec<Section>,
}

fn parse_anchors(sentence: &Sentence) -> Vec<(usize, usize, String)> {
    let mut anchors = vec![];

    for anchor in &sentence.anchors {
        let text = match anchor.text {
            Some(ref text) => text,
            None => &anchor.title,
        };
        let text = regex_replace_all!(r#"('{2,3})"#, text, |_, _| "");
        let text = regex_replace_all!(r#"(\s{2,})"#, &text, |_, _| " ");

        let title = anchor.title.replace(' ', "_");
        let title = capitalize(&title);

        if title.contains("Datoteka:")
            || title.contains("File:")
            || title.contains("datoteka:")
            || title.contains("file:")
        {
            continue;
        }

        let start = sentence.text.find(&*text);
        let start = match start {
            None => continue,
            Some(x) => x,
        };
        let start = sentence.text.as_str()[..start].chars().count();

        let end = start
            + text.chars().count()
            + sentence
                .text
                .chars()
                .skip(start + text.chars().count())
                .position(|char| {
                    char.is_whitespace() || char.is_control() || !char.is_alphanumeric()
                })
                .unwrap_or(0);

        anchors.push((start, end, title));
    }

    anchors
}

fn parse_paragraph(paragraph: &Vec<Sentence>) -> (String, Vec<(usize, usize, String)>) {
    let mut text = String::new();
    let mut anchors: Vec<(usize, usize, String)> = vec![];
    for sentence in paragraph {
        let length = text.chars().count() + if text.is_empty() { 0 } else { 1 };

        anchors.extend(
            parse_anchors(&sentence)
                .into_iter()
                .map(|anchor| (anchor.0 + length, anchor.1 + length, anchor.2)),
        );

        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&sentence.text);
    }

    (text.trim().to_owned(), anchors)
}

#[derive(Debug, ArrowField, ArrowSerialize, ArrowDeserialize)]
struct Anchor {
    start: u32,
    end: u32,
    qid: Option<u32>,
    pageid: Option<u32>,
    title: String,
}

#[derive(Debug, ArrowField, ArrowSerialize, ArrowDeserialize)]
struct DataPoint {
    uuid: String,
    article_title: String,
    article_pageid: u32,
    article_qid: Option<u32>,
    section_heading: Option<String>,
    section_level: u32,
    paragraph_text: String,
    paragraph_anchors: Vec<Anchor>,
}

#[derive(Debug, Deserialize)]
struct MappingRecord {
    title: String,
    pageid: u32,
    qid: Option<u32>,
}

fn write_dataset(split: Vec<DataPoint>, path: &str) {
    let schema = Schema::from(vec![
        Field::new("uuid", DataType::Utf8, false),
        Field::new("article_title", DataType::Utf8, false),
        Field::new("article_pageid", DataType::UInt32, false),
        Field::new("article_qid", DataType::UInt32, true),
        Field::new("section_heading", DataType::Utf8, true),
        Field::new("section_level", DataType::UInt32, false),
        Field::new("paragraph_text", DataType::Utf8, false),
        Field::new(
            "paragraph_anchors",
            DataType::List(Box::new(Field::new(
                "",
                DataType::Struct(vec![
                    Field::new("start", DataType::UInt32, false),
                    Field::new("end", DataType::UInt32, false),
                    Field::new("qid", DataType::UInt32, true),
                    Field::new("pageid", DataType::UInt32, true),
                    Field::new("title", DataType::Utf8, false),
                ]),
                false,
            ))),
            false,
        ),
    ]);

    let options = WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Zstd(None),
        version: Version::V2,
        data_pagesize_limit: None,
    };

    let file = File::create(path).unwrap();

    let mut writer = FileWriter::try_new(file, schema.clone(), options).unwrap();

    let chunks = split
        .iter()
        .array_chunks::<100_000>()
        .map(|examples| {
            let array: Box<dyn Array> = examples.try_into_arrow().unwrap();
            let array = array
                .as_any()
                .downcast_ref::<arrow2::array::StructArray>()
                .unwrap();

            Ok(Chunk::new(vec![array.clone().boxed()]).flatten().unwrap())
        })
        .collect::<Vec<_>>();

    let encodings = schema
        .fields
        .iter()
        .map(|f| transverse(&f.data_type, |_| Encoding::Plain))
        .collect();

    let row_groups =
        RowGroupIterator::try_new(chunks.into_iter(), &schema.clone(), options, encodings).unwrap();

    for group in row_groups {
        writer.write(group.unwrap()).unwrap();
    }

    writer.end(None).unwrap();
}

fn write_statistics(dataset: &Vec<DataPoint>, path: &str) {
    let articles = dataset.iter().map(|x| &x.article_title).unique().count();
    let paragraphs = dataset.len();
    let anchors = dataset
        .par_iter()
        .map(|x| &x.paragraph_anchors)
        .flatten()
        .count();
    let anchors_with_qids = dataset
        .par_iter()
        .map(|x| &x.paragraph_anchors)
        .flatten()
        .filter(|x| x.qid.is_some())
        .count();

    let anchors_with_pageids = dataset
        .par_iter()
        .map(|x| &x.paragraph_anchors)
        .flatten()
        .filter(|x| x.pageid.is_some())
        .count();

    fs::write(
        path,
        format!(
            "{},{},{},{},{}",
            articles, paragraphs, anchors, anchors_with_qids, anchors_with_pageids
        ),
    )
    .unwrap();
}

fn main() {
    let args = Args::parse();

    let mut train = vec![];
    let mut validation = vec![];
    {
        let mut mapping = HashMap::new();
        let reader = File::open(&args.input_wiki2qid).unwrap();

        for record in Reader::new(reader).unwrap() {
            let record = from_value::<MappingRecord>(&record.unwrap()).unwrap();
            mapping.insert(record.title, (record.pageid, record.qid));
        }

        let mapping = Arc::new(mapping);

        let distribution = Arc::new(Bernoulli::new(0.9).unwrap());

        let train = Arc::new(Mutex::new(&mut train));
        let validation = Arc::new(Mutex::new(&mut validation));

        let file = BufReader::new(File::open(&args.input_wiki).unwrap());

        file.lines()
            .par_bridge()
            .filter_map(|line| {
                let article: Article = match serde_json::from_str(&line.unwrap()) {
                    Ok(article) => article,
                    Err(_) => return None,
                };

                let article_title = article.title.replace(' ', "_");

                let article_qid = match mapping.get(&article_title) {
                    Some(&(_, qid)) => qid,
                    None => None,
                };

                let mut examples = vec![];
                for section in article.sections {
                    for paragraph in section.paragraphs {
                        let (text, anchors) = parse_paragraph(&paragraph);

                        if text.is_empty() {
                            continue;
                        }

                        let anchors: Vec<_> = anchors
                            .into_iter()
                            .map(|anchor| {
                                let (pageid, qid) = match mapping.get(&anchor.2) {
                                    Some(&(pageid, qid)) => (Some(pageid), qid),
                                    None => (None, None),
                                };

                                Anchor {
                                    start: anchor.0 as u32,
                                    end: anchor.1 as u32,
                                    qid,
                                    pageid,
                                    title: anchor.2,
                                }
                            })
                            .collect();

                        examples.push(DataPoint {
                            uuid: Uuid::new_v4().to_string(),
                            article_title: article_title.clone(),
                            article_pageid: article.pageid,
                            article_qid,
                            section_heading: match section.title.as_str() {
                                "" => None,
                                x => Some(x.to_string()),
                            },
                            section_level: section.depth,
                            paragraph_text: text,
                            paragraph_anchors: anchors,
                        });
                    }
                }

                if examples.is_empty() {
                    return None;
                }

                Some(examples)
            })
            .flatten()
            .for_each(|example| {
                let mut split = match distribution.sample(&mut thread_rng()) {
                    true => train.lock().unwrap(),
                    false => validation.lock().unwrap(),
                };

                (*split).push(example);
            });
    }

    write_statistics(
        &train,
        Path::new(&args.output_dir)
            .join("train_statistics.csv")
            .to_str()
            .unwrap(),
    );
    write_statistics(
        &validation,
        Path::new(&args.output_dir)
            .join("validation_statistics.csv")
            .to_str()
            .unwrap(),
    );

    write_dataset(
        train,
        Path::new(&args.output_dir)
            .join("train.parquet")
            .to_str()
            .unwrap(),
    );
    write_dataset(
        validation,
        Path::new(&args.output_dir)
            .join("validation.parquet")
            .to_str()
            .unwrap(),
    );
}
