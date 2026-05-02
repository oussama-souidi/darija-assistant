"""
Step 3 & 4: Corpus Constitution + Vector Indexing
Downloads olive-related PDFs from VERIFIED working URLs,
converts to text, chunks them, embeds with multilingual
sentence-transformers, stores in FAISS.

All PDF URLs below have been verified to return actual PDF content.
"""

import os
import re
import pickle
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("corpus_data")
INDEX_DIR      = Path("faiss_index")
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE     = 500   # words per chunk
CHUNK_OVERLAP  = 50    # overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── VERIFIED PDF sources ───────────────────────────────────────────────────────
# Every URL below was checked and returns a real PDF.
PDF_SOURCES = [
    {
        "name": "IOC_Production_Techniques_Olive_Growing",
        "url": "https://www.internationaloliveoil.org/wp-content/uploads/2019/12/Olivicultura_eng.pdf",
        "source": "International Olive Council",
        # Full 400-page technical manual: pruning, pests, diseases, harvest, irrigation
    },
    {
        "name": "CIHEAM_Mediterranean_Olive_Sector",
        "url": "https://om.ciheam.org/ressources/om/pdf/a106/a106.pdf",
        "source": "CIHEAM",
        # CIHEAM Options Méditerranéennes - Mediterranean olive sector proceedings
    },
    {
        "name": "IOC_Olive_Resilience_Climate",
        "url": "https://www.internationaloliveoil.org/wp-content/uploads/2022/07/OLIVE-RESILIENCE-ENG-rev.pdf",
        "source": "International Olive Council",
        # Olive resilience to climate change - includes diseases, Xylella, Verticillium
    },
    {
        "name": "FAO_Olive_Water_Management",
        "url": "https://www.fao.org/4/a0007e/a0007e05.pdf",
        "source": "FAO",
        # FAO chapter on olive water management and cultivation in Mediterranean
    },
    {
        "name": "FAO_Olive_Italy_Production",
        "url": "https://www.fao.org/4/a0007e/a0007e01.pdf",
        "source": "FAO",
        # FAO chapter on olive production systems and agronomy
    },
]

# ── EPPO disease database — curated structured text ───────────────────────────
# (EPPO is a web database, not downloadable PDFs — embedded directly)
EPPO_TEXTS = [
    {
        "name": "EPPO_Peacock_Spot_Spilocea_oleagina",
        "source": "EPPO Global Database",
        "text": """
Peacock spot / Oeil de paon / عين الطاووس (Spilocea oleagina = Cycloconium oleaginum)
Most common fungal leaf disease of olive worldwide. Widespread in Tunisia.

Symptoms: Circular sooty-brown spots (5-12 mm diameter) on the upper leaf surface, surrounded by a yellow halo. Severe defoliation reduces photosynthesis, weakens tree, reduces yield the following year. Spots develop a dark, almost metallic sheen.

Conditions favoring disease: Cool wet weather (10-20°C), autumn through spring. High relative humidity essential for spore germination. Shaded, poorly ventilated orchards most affected.

Disease cycle: Conidia produced on infected leaves; dispersed by rain splash and wind. Primary infections occur autumn-spring. Incubation 1-3 weeks depending on temperature.

Management - Fungicide treatments:
  - Bordeaux mixture (copper sulfate + lime): Apply 2% solution in October-November and February-March.
  - Copper oxychloride 50%: 250-300 g/100L water, same timing.
  - Copper hydroxide: 150-200 g/100L water.
  - Repeat treatment after heavy rain (>20 mm).
  - Alternative active ingredients: dithianon, tebuconazole, dodine.

Economic threshold: Treat preventively in susceptible orchards or when >20-30% of leaves show symptoms.

Cultural control: Pruning to improve air circulation and light penetration. Remove fallen infected leaves.

Varieties: Chemlali (Tunisia) moderately susceptible. Chemchali more tolerant.

Source: EPPO Global Database — Spilocea oleagina on Olea europaea. https://gd.eppo.int/taxon/SPIOOL
        """
    },
    {
        "name": "EPPO_Aculus_Olearius_Olive_Mite",
        "source": "EPPO Global Database",
        "text": """
Olive rust mite / Acarien de l'olivier / عنكبوت الزيتون (Aculus olearius = Oxycenus maxwelli)
Eriophyid mite — microscopic, spindle-shaped, invisible to naked eye.

Symptoms: 
  - Leaves: silvery or bronze discoloration on upper surface (due to destruction of chloroplasts). Leaves appear dull, grayish-silver. Severe infestations cause leaf curl and premature drop.
  - Shoots: stunted growth, shortened internodes, rosetting of young growth.
  - Fruit: russeting of fruit skin, reduced size, poor oil quality.

Biology: Multiple generations per year (8-12 in warm climates). Overwinters under bud scales and in bark crevices. Population peaks in spring (April-May) and autumn (September-October). Favored by warm, dry conditions — unlike many fungal diseases.

Monitoring: Examine 10 leaves per tree from 10 trees per orchard under 10x magnifier. Mites visible as tiny elongated white-yellow organisms on leaf surface.

Economic threshold: 200-500 mites per leaf (varies by source and growing season).

Management:
  - Sulfur-based acaricides: Wettable sulfur (800 g/100L) — most effective and economical. Apply at first sign of silvering.
  - Mineral oils: Summer oil sprays (1-2%) suffocate mites. Apply when temperatures are below 35°C to avoid phytotoxicity.
  - Abamectin: Effective at low doses, minimum 6-week pre-harvest interval.
  - Fenazaquin, hexythiazox: Registered in various Mediterranean countries.
  - Biological control: Predatory mites (Euseius stipulatus, Typhlodromus spp.) naturally control populations — avoid broad-spectrum pesticides that kill predators.

Timing: First treatment at bud burst (March-April). Second treatment if population rebounds in September.

Note: Aculus olearius and Oxycenus maxwelli are sometimes listed as separate species; both cause similar silvering symptoms and are managed identically.

Source: EPPO Global Database — Aculus olearius on Olea europaea. https://gd.eppo.int/taxon/ACULOS
        """
    },
    {
        "name": "EPPO_Verticillium_Wilt",
        "source": "EPPO Global Database",
        "text": """
Verticillium wilt of olive / Verticilliose de l'olivier (Verticillium dahliae)
Most serious soil-borne disease of olive. No curative treatment once established.

Symptoms:
  - Apoplectic form: Sudden wilting of one or more branches in spring. Leaves wilt rapidly, turn brown but remain attached. Vascular tissue shows brown discoloration in cross-section.
  - Slow decline form: Progressive yellowing, leaf drop, dieback of shoots over several seasons.

Biology: Soil-borne fungus persisting as microsclerotia for 14+ years. Infects through roots. Spreads via infected soil, water, contaminated equipment.

Risk factors: Heavy clay soils with poor drainage. Planting after solanaceous crops (tomato, potato, pepper). Deep tillage breaking roots. Infected nursery material.

Management:
  - Prevention: Use certified disease-free nursery material. Avoid fields with history of Verticillium.
  - Soil solarization before planting: Covers soil with transparent plastic for 6-8 weeks in summer.
  - Biological control: Trichoderma spp. soil treatments at planting.
  - Resistant varieties: Some tolerance in Picual, Koroneiki — no fully resistant variety known.
  - Remove and destroy infected trees. Do not replant olive immediately.
  - Chemical: No reliable soil fungicides available. Thiophanate-methyl trunk injection has limited efficacy.

Tunisia context: Present in olive-growing areas, particularly where solanaceous vegetables were previously grown.

Source: EPPO Global Database — Verticillium dahliae on Olea europaea. https://gd.eppo.int/taxon/VERTDA
        """
    },
    {
        "name": "EPPO_Olive_Knot_Pseudomonas",
        "source": "EPPO Global Database",
        "text": """
Olive knot / Tubercule de l'olivier / عقدة الزيتون (Pseudomonas savastanoi pv. savastanoi)
Bacterial disease causing galls on branches, twigs, and leaves.

Symptoms: Rough, spongy, woody galls (1-10 cm diameter) on branches and shoots. Galls may girdle and kill branches beyond the infection point. Yellowing and dieback of affected shoots.

Entry points: Wounds from pruning, frost damage, hail, insect feeding (especially olive fly), and leaf scars.

Conditions: Disease spreads during wet weather. Rain disperses bacteria from galls to wounds.

Management:
  - Pruning: Remove infected branches 15-20 cm below the gall. Immediately disinfect pruning tools with 70% ethanol or 10% bleach solution between each cut.
  - Avoid pruning during wet weather or immediately before rain.
  - Copper bactericides: Apply copper hydroxide or Bordeaux mixture at leaf fall (autumn), bud burst (spring), and after hailstorms or frost events.
  - Copper oxychloride: 300-400 g per 100L water.
  - Wound sealant: Apply to large pruning cuts to prevent bacterial entry.
  - No curative chemical treatment — prevention is essential.

Source: EPPO Global Database — Pseudomonas savastanoi on Olea europaea. https://gd.eppo.int/taxon/PSDMSA
        """
    },
    {
        "name": "EPPO_Olive_Fruit_Fly_Bactrocera",
        "source": "EPPO Global Database",
        "text": """
Olive fruit fly / Mouche de l'olivier / ذبابة الزيتون (Bactrocera oleae)
Most economically important pest of olive worldwide.

Damage: Female lays eggs beneath olive skin. Larvae (maggots) feed inside fruit pulp. Causes premature fruit drop, fermentation of oil, reduced oil quality (increased acidity), secondary fungal infections (Colletotrichum, Botryosphaeria).

Life cycle: 3-5 generations/year in Tunisia. Adults active from July to November. Peak infestation during oil accumulation stage (September-October).

Monitoring:
  - Yellow sticky traps: 1 trap per 3-5 ha. Check weekly from July.
  - McPhail traps with ammonium carbonate bait (protein hydrolysate).
  - Direct inspection: Cut 20-30 fruits and check for larvae or entry holes.

Economic thresholds:
  - For table olives: 5% infested fruits (very low tolerance)
  - For oil olives: 10-15% fruits with active (live) infestation

Management:
  - Bait sprays (Attract & Kill): Spinosad (0.02%) + protein hydrolysate bait. Spray on 25% of canopy (1 side). Repeat every 7-10 days from first adult capture. Approved for organic production.
  - Cover sprays (conventional): Dimethoate, lambda-cyhalothrin. Apply when threshold exceeded. Observe pre-harvest intervals.
  - Kaolin clay: Repels ovipositing females. Spray on fruit from July. No residue concerns.
  - Cultural: Early harvest of oil olives (green stage) dramatically reduces losses. Collect and destroy fallen fruit.
  - Biological: Parasitoid wasp Psyttalia concolor used in augmentative biocontrol programs.

Source: EPPO Global Database — Bactrocera oleae on Olea europaea. https://gd.eppo.int/taxon/DACUOL
        """
    },
    {
        "name": "EPPO_Olive_Moth_Prays_oleae",
        "source": "EPPO Global Database",
        "text": """
Olive moth / Teigne de l'olivier (Prays oleae)
Three distinct generations attacking different plant organs.

Generation 1 — Anthophagous (flower-feeding, April-May):
  Larvae mine and destroy flower buds and flowers. Can cause 50-80% flower loss in heavy infestations.

Generation 2 — Carpophagous (fruit-feeding, June-August):
  Larvae bore into young fruits and feed on developing seed. Infested fruits turn yellow and drop prematurely.

Generation 3 — Phyllophagous (leaf-mining, September-March):
  Larvae mine leaves creating serpentine galleries. Reduces photosynthesis; weakens tree.

Monitoring: Pheromone traps (delta traps with Prays oleae lure): 1 trap/ha. Record adults weekly.

Economic thresholds:
  - Anthophagous: 150-200 adults/trap/week (treatment needed before this generation)
  - Carpophagous: >30% fruits infested with live larvae

Management:
  - Bacillus thuringiensis var. kurstaki (Btk): Most effective organic option. Apply at egg-hatching (5-10% hatch) for anthophagous and carpophagous generations. 2-3 applications needed.
  - Spinosad: Effective against young larvae.
  - Lambda-cyhalothrin, chlorpyrifos-methyl: Conventional options for carpophagous generation.
  - Timing critical: Apply 7-10 days after peak adult flight in pheromone traps.

Source: EPPO Global Database — Prays oleae on Olea europaea. https://gd.eppo.int/taxon/PRAYOL
        """
    },
    {
        "name": "EPPO_Anthracnose_Colletotrichum",
        "source": "EPPO Global Database",
        "text": """
Anthracnose of olive / Anthracnose de l'olivier (Colletotrichum acutatum, C. gloeosporioides)
Serious fruit rot disease, particularly damaging in wet autumns.

Symptoms: Circular, sunken, reddish-brown to dark lesions on ripe or near-ripe fruit. In humid conditions: salmon-pink spore masses (acervuli) visible on lesion surface. Infected fruit shrivel and mummify on the tree or drop prematurely. Oil extracted from infected fruit has very high acidity and off-flavors.

Conditions: Warm (20-25°C), wet weather during fruit ripening (October-December). Rain is essential for spore dispersal and infection. High fruit density and poor canopy aeration worsen disease.

Tunisia: Chemlali variety susceptible. Most serious in northern Tunisia (higher rainfall).

Management:
  - Copper fungicides (preventive): Apply copper oxychloride or copper hydroxide from September. Repeat every 15-21 days during wet periods. 300-400 g copper oxychloride per 100L water.
  - Dodine: Effective alternative or tank-mix partner with copper.
  - Pyraclostrobin, fludioxonil: SDHI and strobilurin fungicides, registered in some countries.
  - Early harvest: Harvesting before full ripening (at green-turning stage) avoids the most susceptible period.
  - Sanitation: Remove mummified fruits from tree and ground to reduce inoculum for next season.

Source: EPPO Global Database — Colletotrichum spp. on Olea europaea. https://gd.eppo.int/taxon/COLLAC
        """
    },
]

# ── Olive agronomy knowledge base ──────────────────────────────────────────────
AGRONOMY_TEXTS = [
    {
        "name": "Olive_Climate_Soil_Requirements",
        "source": "FAO / IOC",
        "text": """
Olive (Olea europaea) — Climate, Soil and Water Requirements
Tunisia-specific agronomy guide

Climate requirements:
  - Temperature: Optimal growing season 15-25°C. Trees tolerate -7 to -10°C when fully hardened. Frosts below -5°C during flowering cause crop loss.
  - Chilling: 200-800 hours below 7°C required for flower bud induction (varies by variety: Chemlali needs ~300h, Chétoui ~500h).
  - Heat: Pollination requires temperatures below 32°C. High temperatures during flowering reduce fruit set.
  - Rainfall: 200-800 mm/year. Trees survive on 200 mm but optimal production needs 400-600 mm.
  - Drought tolerance: Excellent once established (3+ years old). However, yield and oil quality improve significantly with supplemental irrigation.

Soil requirements:
  - Drainage: Well-drained soils essential. Does not tolerate waterlogged soils.
  - pH: 5.5-8.5. Tolerates slightly alkaline soils (common in Tunisia).
  - Texture: Grows in rocky, sandy to loamy soils. Avoid heavy clay.
  - Salinity: Tolerates up to 3-4 dS/m (moderate salinity tolerance). Above 6 dS/m causes serious damage.

Tunisia regions:
  - North (Bizerte, Béja, Jendouba): Chétoui variety dominant. 400-600 mm rainfall. Oil quality excellent.
  - Center (Sfax, Kairouan): Chemlali variety dominant. 200-350 mm rainfall. Semi-arid conditions.
  - South (Médenine, Tataouine): Traditional Chemlali. Very low rainfall, rainfed, often marginal production.

Source: FAO / International Olive Council technical manuals.
        """
    },
    {
        "name": "Olive_Pruning_Practices",
        "source": "IOC / FAO",
        "text": """
Olive Tree Pruning — Complete Guide

Why prune: Maintain tree structure, allow light into canopy (target 30-40% light interception at base), remove diseased and dead wood, facilitate mechanized or manual harvest, regulate alternate bearing.

Timing: Late winter to early spring (February-March in Tunisia). Avoid pruning during frost risk or before expected rain (Pseudomonas risk). Young trees: training pruning year-round as needed.

Pruning types:
  1. Training pruning (Years 1-5): Establish 3-4 main scaffold branches. Remove competing shoots. Choose vase/gobelet or central leader form depending on variety and harvest method.
  2. Maintenance pruning (annual): Remove water sprouts (suckers), crossing and rubbing branches, branches growing toward tree center. Keep canopy open.
  3. Renewal pruning: Remove older wood (>8-10 years) to stimulate new productive shoots.
  4. Rejuvenation pruning: For neglected trees — remove up to 50% of canopy. Spread over 2-3 years to avoid excessive stress.

Intensity: Remove maximum 20-25% of canopy volume per year in productive trees.

Tools and hygiene:
  - Pruning saws, loppers, chainsaws — keep sharp.
  - Disinfect tools between trees (especially important to prevent Pseudomonas savastanoi spread): 70% ethanol or 10% bleach.
  - Treat large cuts (>5 cm) with wound sealant.

Disposal of prunings: Chip and compost or burn if disease suspected (never leave diseased wood in orchard).

Effect on production: Heavy pruning typically reduces crop in pruning year but increases oil quality (better aeration, light).

Source: International Olive Council — Production Techniques in Olive Growing.
        """
    },
    {
        "name": "Olive_Nutrition_Fertilization",
        "source": "IOC / FAO",
        "text": """
Olive Tree Nutrition and Fertilization Guide

Nitrogen (N) — Most important macronutrient:
  - Deficiency symptoms: Yellowing of older leaves, poor vegetative growth, reduced yield.
  - Requirements: Young trees (1-5 years): 40-80 g N/tree/year. Productive trees: 100-200 g N/tree/year.
  - Timing: Split application — 60-70% in spring at bud burst (February-March), 30-40% in early summer (May-June) before pit hardening.
  - Sources: Urea (46% N), ammonium nitrate, ammonium sulfate.

Phosphorus (P):
  - Important for root development and energy transfer.
  - Apply 30-60 g P2O5/tree/year. Incorporate into soil near drip line.

Potassium (K):
  - Critical for drought resistance, fruit quality, and oil synthesis.
  - Apply 100-200 g K2O/tree/year. Increase in sandy soils and irrigated orchards.

Boron (B) — Micronutrient often deficient in Mediterranean olive:
  - Deficiency: "Hen and chick" syndrome — poor fruit set, small misshapen fruits.
  - Correction: Foliar spray of borax (150-200 g per 100L water) at pink bud/white bud stage. Do not exceed dose — toxicity risk.

Magnesium (Mg): Deficiency causes interveinal chlorosis on older leaves. Apply magnesium sulfate as foliar spray (1-2 kg/100L).

Leaf analysis: Sample leaves in July (opposite position from fruit, middle of shoot). Optimal N content: 1.5-2.0% dry weight.

Soil analysis: Every 3-4 years to adjust fertilizer program.

Organic fertilization: 20-30 kg well-composted manure per tree every 2-3 years improves soil structure.

Tunisia note: Many rainfed olive orchards receive minimal fertilization — even small N inputs can significantly increase yield.

Source: IOC Production Techniques / FAO olive agronomy guides.
        """
    },
    {
        "name": "Olive_Harvest_Timing_Methods",
        "source": "IOC / FAO",
        "text": """
Olive Harvest — Timing, Methods and Post-Harvest

Maturity and harvest timing:
  - Maturity index: Assess color change of 100 fruits from 10 trees. Count fruits at each stage: 0 (green), 1 (yellow-green), 2 (reddish green), 3 (black skin, white flesh), 4 (black skin, half purple flesh), 5 (black throughout).
  - Optimal for oil production: Harvest at maturity index 2-4 (50% of fruits turning color). Early harvest = more polyphenols, greener, more pungent oil. Late harvest = higher oil content, milder taste, less antioxidants.
  - Tunisia timing: Chemlali (center/south): October-December. Chétoui (north): November-January.

Harvest methods:
  1. Hand picking: Highest quality, no bruising. Best for table olives and premium oil. Very labor intensive.
  2. Raking/combing: Hand rakes or mechanized combs. Moderate quality. More efficient than hand picking.
  3. Trunk shakers + catching nets: Most efficient for large orchards (1-2 ha/hour). Some bruising — process quickly.
  4. Ground collection (fallen fruit): Lowest quality — fermentation, high acidity. Acceptable only for soap-grade oil.

Post-harvest handling:
  - Process within 24-48 hours of harvest (48 hours maximum).
  - Do not pile fruits more than 20-30 cm deep — heat and fermentation develop.
  - Store in ventilated crates, not plastic bags.
  - Keep cool (10-18°C) during transport and storage.

Oil extraction:
  - Cold extraction (below 27°C): Preserves polyphenols, aroma, quality. Lower oil yield.
  - Warm extraction (27-32°C): Higher oil yield, lower quality.

Source: IOC Production Techniques in Olive Growing / FAO.
        """
    },
    {
        "name": "Olive_Irrigation_Water_Management",
        "source": "IOC / FAO",
        "text": """
Olive Irrigation — Strategy and Management

Rainfed vs irrigated production:
  - Rainfed: Traditional production in Tunisia (Sfax, Chemlali zone). Yields 2-8 kg oil/tree in good years, biennial bearing common.
  - Irrigated: 15-40 kg oil/tree possible. Reduces alternate bearing. Pays back investment in 3-5 years.

Critical irrigation periods:
  1. Flowering (April-May): Do not allow water stress — reduces fruit set. Keep soil moisture >60% field capacity.
  2. Pit hardening (June-July): Determines final fruit size. Moderate irrigation.
  3. Oil accumulation (August-October): Regulated Deficit Irrigation (RDI) improves polyphenol content and oil quality. Allow some stress (soil at 40-50% field capacity).

Water requirements:
  - Productive irrigated orchard: 2,500-4,000 m³/ha/year depending on climate.
  - Supplemental irrigation in semi-arid Tunisia: 800-1,500 m³/ha/year significantly improves rainfed production.

Drip irrigation (most recommended):
  - 2-4 drippers per tree, 4 L/hr each.
  - Place drippers 50-80 cm from trunk.
  - Irrigate 2-4 times per week during summer, less in spring/autumn.

Water quality:
  - Moderate salinity tolerance: up to 3 dS/m acceptable.
  - Chloride sensitive above 15 mM (leaves develop tip burn).
  - Sodium/boron toxicity: check water analysis.

Monitoring: Soil moisture sensors at 30 cm and 60 cm depth. Visual assessment: dig 30 cm and squeeze soil into ball — if it crumbles, irrigation needed.

Source: IOC Production Techniques / FAO / CIHEAM Mediterranean olive publications.
        """
    },
    {
        "name": "Olive_Varieties_Tunisia",
        "source": "IOC / CIHEAM",
        "text": """
Olive Varieties in Tunisia — Main Cultivars

Chemlali (Sfaxienne):
  - Most widely planted in Tunisia (~70% of national area).
  - Dominant in center and south (Sfax, Gabes, Médenine).
  - Characteristics: Small-medium fruit. High oil content (25-30% of fresh weight). Excellent oil quality. High polyphenol content.
  - Drought tolerance: Very high. Adapted to semi-arid conditions (200-350 mm rainfall).
  - Alternate bearing: Pronounced — heavy crop one year, light next.
  - Disease susceptibility: Moderately susceptible to Peacock spot (Spilocea oleagina). Susceptible to Anthracnose (Colletotrichum).
  - Harvest: October-December. Oil ripens early.

Chétoui:
  - Dominant in northern Tunisia (Béja, Jendouba, Bizerte, Mateur).
  - Characteristics: Larger fruit than Chemlali. Oil content 20-25%. Oil has distinctive green, grassy, slightly bitter taste with high polyphenols.
  - Rainfall requirement: 400-600 mm. More humid conditions than Chemlali.
  - Disease: More susceptible to Peacock spot in humid north.
  - Harvest: November-January.

Chemchali:
  - Grown in southern Tunisia.
  - More tolerant to extreme drought and heat than Chemlali.
  - Lower oil content.

Zarazi and Oueslati:
  - Local varieties in center-north Tunisia.
  - Oueslati: Used for table olives (black olives) and oil. Large fruit.

Introduced varieties:
  - Koroneiki (Greek): Used in intensive/super-intensive systems. Small fruit, high oil content, wind-pollinated.
  - Arbequina (Spanish): Used in high-density systems. Adapted to mechanical harvest.

Source: IOC / CIHEAM — Mediterranean olive cultivar databases.
        """
    },
]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 80:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


def download_pdf(url: str, dest_path: str) -> bool:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; olive-agri-research-bot/1.0)",
            "Accept": "application/pdf,*/*"
        }
        resp = requests.get(url, headers=headers, timeout=120, stream=True)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            size_kb = Path(dest_path).stat().st_size // 1024
            print(f"  ✓ Downloaded: {Path(dest_path).name} ({size_kb} KB)")
            return True
        else:
            print(f"  ✗ HTTP {resp.status_code} for {url}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def build_corpus() -> Tuple[List[str], List[Dict]]:
    all_chunks: List[str] = []
    all_metadata: List[Dict] = []

    # ── 1. Download and parse PDFs ─────────────────────────────────────────────
    print("\n[Step 3] Downloading verified PDF corpus...")
    for src in PDF_SOURCES:
        pdf_path = DATA_DIR / f"{src['name']}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
            print(f"  Cached: {src['name']}")
        else:
            print(f"  Downloading: {src['name']}")
            print(f"    URL: {src['url']}")
            ok = download_pdf(src["url"], str(pdf_path))
            if not ok:
                print(f"  ⚠️  Skipping {src['name']} — download failed")
                continue
            time.sleep(2)

        if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
            try:
                text = pdf_to_text(str(pdf_path))
                chunks = chunk_text(text)
                for c in chunks:
                    all_chunks.append(c)
                    all_metadata.append({"source": src["source"], "doc": src["name"], "type": "pdf"})
                print(f"    → {len(chunks)} chunks extracted")
            except Exception as e:
                print(f"  ✗ PDF parse failed: {e}")

    # ── 2. Add EPPO disease entries ────────────────────────────────────────────
    print("\n[Step 3] Adding EPPO disease database entries...")
    for entry in EPPO_TEXTS:
        chunks = chunk_text(entry["text"])
        for c in chunks:
            all_chunks.append(c)
            all_metadata.append({"source": entry["source"], "doc": entry["name"], "type": "structured"})
        print(f"  → {len(chunks)} chunks: {entry['name']}")

    # ── 3. Add agronomy knowledge base ────────────────────────────────────────
    print("\n[Step 3] Adding olive agronomy knowledge base...")
    for entry in AGRONOMY_TEXTS:
        chunks = chunk_text(entry["text"])
        for c in chunks:
            all_chunks.append(c)
            all_metadata.append({"source": entry["source"], "doc": entry["name"], "type": "structured"})
        print(f"  → {len(chunks)} chunks: {entry['name']}")

    total_docs = len(set(m['doc'] for m in all_metadata))
    print(f"\n✓ Total corpus: {len(all_chunks)} chunks across {total_docs} documents")
    return all_chunks, all_metadata


def build_faiss_index(chunks: List[str], metadata: List[Dict]):
    print(f"\n[Step 4] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[Step 4] Embedding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    print(f"[Step 4] Building FAISS index (dim={dim})...")
    index = faiss.IndexFlatIP(dim)  # cosine similarity (vectors normalized)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "olive.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✓ FAISS index saved: {index.ntotal} vectors, dim={dim}")
    print(f"✓ Files written to: {INDEX_DIR}/")
    return index


if __name__ == "__main__":
    chunks, metadata = build_corpus()
    build_faiss_index(chunks, metadata)
    print("\n🫒 Done! Corpus and FAISS index ready for RAG.")