import sqlite3
import os
from bs4 import BeautifulSoup, Tag
from joblib import Parallel, delayed
from pathlib import Path
from selenium import webdriver
from tqdm import tqdm

DB_NAME = "charts.db"
HTML_FILES = {"gamerch": "cached_html/gamerch"}
Path("cached_html").mkdir(exist_ok=True)
[Path(dir).mkdir(exist_ok=True) for dir in HTML_FILES.values()]
[Path(dir+"/category").mkdir(exist_ok=True) for dir in HTML_FILES.values()]
[Path(dir+"/song").mkdir(exist_ok=True) for dir in HTML_FILES.values()]

if not os.path.exists(DB_NAME):
    db = sqlite3.connect(DB_NAME, isolation_level=None)
    db.execute(
    '''CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY,
            name VARCHAR(50),
            artist VARCHAR(50),
            link VARCHAR(200),
            category VARCHAR(24),
            bpm INTEGER(4),
            jp_only INTEGER(1)
        )'''
    )
    db.execute(
    '''CREATE TABLE IF NOT EXISTS charts (
            id INTEGER PRIMARY KEY,
            difficulty VARCHAR(20),
            level VARCHAR(5),
            const VARCHAR(5),
            combo INTEGER,
            taps INTEGER,
            holds INTEGER,
            slides INTEGER,
            touchs INTEGER,
            breaks INTEGER,
            version VARCHAR(10),
            song_id INTEGER NOT NULL,
            FOREIGN KEY(song_id) REFERENCES songs(id)
        )'''
    )
    db.close()

OPTION = webdriver.FirefoxOptions()
CACHE_SITES = True
OPTION.add_argument("--headless")

class Sites(object):
    def __init__(self, gamerch:str, sega_list:str, category:str) -> None:
        self.gamerch = gamerch
        self.sega_list = sega_list
        self.category = category

    def get_songs(self):
        if os.path.exists((html_file := f"{HTML_FILES['gamerch']}/category/{self.category}.html")) and CACHE_SITES:
            with open(html_file, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, 'html.parser')
        else:
            driver = webdriver.Firefox(OPTION)
            driver.install_addon("ublock.xpi")
            driver.get(self.gamerch)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            with open(html_file, "w", encoding="utf-8") as file:
                file.write(str(soup))
            driver.close()
        
        songs :list[Tag] = soup.find_all("div", class_="mu__table")

        if len(songs) == 1:
            songs   :list[Tag] = songs[0].find_all("tr")[2:]
            
            names   :list[str] = [song.find_next("th").text for song in songs]
            links   :list[str] = [song.find_next("a")["href"] for song in songs]
            artists :list[str] = [song.find_next("td").text for song in songs]
            bpms    :list[str] = [song.find_all("td")[-1].text for song in songs]
        elif self.category == "Deleted":
            songs   :list[Tag] = [tr for song in songs for tr in song.find_all("tr")[3:]]
            songs   :list[Tag] = [song for song in songs if len(song.find_all("td")) > 2]

            names   :list[str] = [song.find_next("td").text for song in songs]
            links   :list[str] = [song.find_next("a")["href"] for song in songs]
            artists :list[str] = [song.find_all("td")[1].text for song in songs]
            bpms    :list[str] = [song.find_all("td")[-1].text for song in songs]
        else:
            raise Exception("Unknown Soup Format")
        
        return [song + (self.category,)  for song in zip(names, links, artists, bpms) if song[-1].isdigit()]

with open("Problematic Songs.txt", "w", encoding='utf-8'): pass

DIFFUCULTIES = {
    'text-align:center;background-color:#00ced1' : "EASY",
    'text-align:center;background-color:#98fb98' : 'BASIC',
    'text-align:center;background-color:#ffa500' : 'ADVANCED',
    'text-align:center;background-color:#fa8080' : 'EXPERT',
    'text-align:center;background-color:#ee82ee' : 'MASTER',
    'text-align:center;background-color:#ffceff' : 'Re:MASTER',
    'text-align:center;background-color:#ff5296' : "UTAGE",
}

VERSION = {
    'EASY' : "ST",
    'BASIC' : 'DX',
    'UTAGE' : 'DX',
}

class Song(object):   
    def __init__(self, id, name, artist, link, bpm, category) -> None:
        self.id = id
        self.name = name
        self.artist = artist
        self.link = link
        self.bpm = bpm
        self.category = category

    def get_chart_info(self, driver: webdriver.Firefox) -> list[tuple[str]]:
        driver.get(self.link)
        soup :Tag= BeautifulSoup(driver.page_source, 'html.parser').find_all("div", class_="main")[0]

        tables :list[Tag] = soup.find_all(["h4", "h3"],string=["スタンダード譜面", "譜面データ", "でらっくす譜面"])
        assert len(tables) > 0

        tables = set(t.find_next_sibling("div", class_="mu__table") for t in tables)

        def process_table(table:Tag):
            header1 :list[Tag] = [h for h in [row.text for row in table.find_all("tr")[0]] if h not in ["内訳", "スコア"]]
            header2 :list[Tag] = [h for h in [row.find_next("span").text for row in table.find_all("tr")[1]] if h not in ["SSS", "SSS+"]]
            header = {h:i for i,h in enumerate(header1 + header2)}
            rows    :list[list[Tag]] = [row.find_all(["th","td"]) for row in table.find_all("tr")[2:]]
            
            if len(rows[0]) != len(rows[-1]): rows = rows[:-1]

            diffs   :list[str] = [DIFFUCULTIES.get(row[header.get("Lv")]["style"]) for row in rows]
            levels  :list[str] = [row[header.get("Lv")].text for row in rows]
            consts  :list[str] = [row[idx].text for row in rows] if (idx := header.get("定数",None)) else ["-"] * len(rows)
            combos  :list[str] = [row[header.get("総数")].text for row in rows]
            taps    :list[str] = [row[header.get("Tap")].text for row in rows]
            holds   :list[str] = [row[header.get("Hold")].text for row in rows]
            slides  :list[str] = [row[header.get("Slide")].text for row in rows]
            touches :list[str] = [row[idx].text for row in rows] if (idx := header.get("Touch",None)) else ["0"] * len(rows)
            breaks  :list[str] = [row[header.get("Break")].text for row in rows]
            version :list[str] = [VERSION.get(diffs[0])] * len(rows)
            ids     :list[str] = [self.id] * len(rows)
            
            return zip(diffs,levels,consts,combos,taps,holds,slides,touches,breaks,version,ids,version,diffs,ids)

        return [chart for table in tables for chart in process_table(table)]
        

table_links = [
    Sites("https://gamerch.com/maimai/entry/533381", "pop_anime", "POPS & ANIME"),
    Sites("https://gamerch.com/maimai/entry/533382", "niconico", "niconico&VOCALOID"),
    Sites("https://gamerch.com/maimai/entry/533383", "toho", "東方Project"),
    Sites("https://gamerch.com/maimai/entry/533385", "variety", "GAME&VARIETY"),
    Sites("https://gamerch.com/maimai/entry/533386", "maimai", "maimai"),
    Sites("https://gamerch.com/maimai/entry/533825", "gekichu", "オンゲキ&CHUNITHM"),
    Sites("https://gamerch.com/maimai/entry/533442", None, "Deleted"),
]

def get_songs(sites:list[Sites]) -> list[tuple[str]]:
    # songs = [site.get_songs() for site in sites]
    songs = Parallel(-1)(delayed(site.get_songs)() for site in sites)
    deleted = [song[0] for song in songs[-1]]
    return [s for song in songs[:-1] for s in song if s[0] not in deleted]

def get_all_songs() -> list[Song]:
    db = sqlite3.connect(DB_NAME, isolation_level=None)

    print("Retrieve songs list")
    songs = get_songs(table_links)
    songs = [s + (s[0],) for s in songs]

    statement = '''
    INSERT INTO songs (name,link,artist,category,bpm) SELECT ?,?,?,?,?
    WHERE NOT EXISTS(
    SELECT name FROM songs WHERE 
        name = ?
    )'''
    db.executemany(statement, (songs))

    statement = '''SELECT * FROM songs'''
    songs = [Song(id, name, artist, link, bpm, category) for id, name, artist, link, bpm, category, _ in db.execute(statement)]

    db.close()
    return songs

def get_chart_info(songs:list[Song]):
    db = sqlite3.connect(DB_NAME, isolation_level=None)

    db.execute("PRAGMA foreign_keys = ON")
    
    statement = '''SELECT DISTINCT song_id FROM charts '''
    can_skip_id = [id[0] for id in db.execute(statement)]
    songs = [song for song in songs if song.id not in can_skip_id]
    db.close()
    
    statement = '''
        INSERT INTO charts (difficulty, level, const, combo, taps, holds, slides, touchs, breaks, version, song_id) 
        SELECT ?,?,?,?,?,?,?,?,?,?,?
        WHERE NOT EXISTS(
        SELECT version, difficulty, song_id FROM charts WHERE
            version=? AND difficulty=? AND song_id=?
        )
    '''

    def write_to_db(songs:list[Song], pb:tqdm):
        driver = webdriver.Firefox(OPTION)
        driver.install_addon("ublock.xpi")

        def scrape(song:Song):
            try:
                charts = song.get_chart_info(driver)

                db = sqlite3.connect(DB_NAME, isolation_level=None)
                db.executemany(statement, charts)
                db.close()
            except Exception as e:
                with open("Problematic Songs.txt", "a", encoding='utf-8') as file:
                    file.write(f"{song.name} : {song.link}\n")
                pb.failed += 1
                pb.set_description(f"{pb.failed} failed")
                raise e
            finally:    
                pb.update(1)

        [scrape(song) for song in songs]
        driver.close()

    def chunks(lst:list, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    if len(songs) == 0: return
    
    with tqdm(total=len(songs), desc="Retrieve charts") as pb:
        pb.failed = 0
        # write_to_db(songs,pb)
        import psutil
        from math import ceil
        threads = psutil.cpu_count()
        songs = list(chunks(songs, ceil(len(songs) / threads)))
        Parallel(threads,require="sharedmem")(delayed(write_to_db)(song,pb) for song in songs)
    
if __name__ == "__main__":
    get_chart_info(get_all_songs())
