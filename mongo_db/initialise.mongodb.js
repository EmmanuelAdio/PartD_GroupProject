/* global use, db */
// MongoDB Playground (VS Code)

// 1) Select your project database
use('partd_group');

// 2) ----- INTENTS COLLECTION -----
// Minimal structure: _id, domain, patterns[{regex, flags}]
const intents = require('./intent_patterns.json');

intents.forEach(doc => {
  db.getCollection('intents').replaceOne({ _id: doc._id }, doc, { upsert: true });
});


// 3) ----- GAZETTEERS COLLECTION -----
const gazetteers = require('./gazetteer.json');

// build aliases_flat for each gazetteer
gazetteers.forEach(g => {
  const flat = [];
  g.items.forEach(item => item.aliases.forEach(a => flat.push(a.toLowerCase())));
  g.aliases_flat = Array.from(new Set(flat)).sort();

  db.getCollection('gazetteers').replaceOne({ _id: g._id }, g, { upsert: true });
});


// 4) Create an index to speed alias lookup (safe to re-run)
db.getCollection('gazetteers').createIndex({ aliases_flat: 1 });


// 5) Quick sanity check: show inserted docs
const intentCount = db.getCollection('intents').countDocuments();
const gazCount = db.getCollection('gazetteers').countDocuments();


db.getCollection('intents').find({}, { _id: 1, domain: 1 }).limit(20);
db.getCollection('gazetteers').find({}, { _id: 1, "items.canonical": 1 }).limit(20);

console.log(`Seeded DB=partd_group | intents=${intentCount} | gazetteers=${gazCount}`);
console.log("Initialization complete.");

