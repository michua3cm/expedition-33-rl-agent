--[[
  UE4SS Lua Mod: StateReader
  ===========================
  Reads Expedition 33 battle state from the UE5 reflection system every
  tick and writes it as JSON to a temp file.  The Python RL agent reads
  this file via environment.ue4ss_reader.StateReader.

  Installation
  ------------
  1. Install UE4SS (Nexus Mods mod 630) into:
       .../Expedition 33/Sandfall/Binaries/Win64/
  2. Copy this entire 'StateReader' folder to:
       .../Win64/Mods/StateReader/
  3. In .../Win64/Mods/mods.txt, add the line:
       StateReader : 1
  4. Launch the game.  The JSON file will appear in %TEMP% on first battle tick.

  Configuration — REQUIRED
  -------------------------
  Before the reader will return real values, fill in the CONFIG table below
  with the actual Blueprint class names and property names.  Discover them:

    a) Launch the game and enter a battle.
    b) Open the UE4SS console (F10 → UE4SS tab → Live Property Viewer).
    c) Search for classes containing "Battle", "Character", "Enemy".
    d) Expand the matching live instances to read their property names and values.
    e) Replace each "TODO" string below with the discovered name.

  Output file
  -----------
  Writes one JSON object per tick to OUTPUT_FILE (default: %TEMP%\expedition33_state.json).
  The Python reader expects this exact schema:
    {
      "player_hp": <number>,   "player_hp_max": <number>,
      "enemy_hp":  <number>,   "enemy_hp_max":  <number>,
      "player_ap": <integer>,
      "enemy_break": <number>, "enemy_break_max": <number>,
      "in_battle": <bool>,     "is_offensive_phase": <bool>
    }
--]]

-- ── Configuration ─────────────────────────────────────────────────────────────
local CONFIG = {
    -- Player character Blueprint class (from UE4SS Live Property Viewer)
    player_class            = "TODO",   -- e.g. "BP_BattleCharacter_C"
    player_hp_prop          = "TODO",   -- e.g. "CurrentHP"
    player_hp_max_prop      = "TODO",   -- e.g. "MaxHP"
    player_ap_prop          = "TODO",   -- Action Points, integer 0–9

    -- Enemy Blueprint class
    enemy_class             = "TODO",   -- e.g. "BP_EnemyBase_C"
    enemy_hp_prop           = "TODO",
    enemy_hp_max_prop       = "TODO",
    enemy_break_prop        = "TODO",   -- break / stun meter
    enemy_break_max_prop    = "TODO",

    -- Battle controller Blueprint class
    battle_class            = "TODO",   -- e.g. "BP_BattleController_C"
    in_battle_prop          = "TODO",   -- bool property
    is_offensive_phase_prop = "TODO",   -- bool: true = player's action turn

    -- Write rate (max writes per second; 0 = write every game tick)
    write_hz = 20,
}

-- ── Output file path ──────────────────────────────────────────────────────────
local function get_output_path()
    local tmp = os.getenv("TEMP") or os.getenv("TMPDIR") or "C:\\Windows\\Temp"
    return tmp .. "\\expedition33_state.json"
end

local OUTPUT_FILE = get_output_path()

-- ── Helpers ───────────────────────────────────────────────────────────────────
local function safe_read(obj, prop)
    if obj == nil then return nil end
    local ok, val = pcall(function() return obj[prop] end)
    return ok and val or nil
end

local function bool_to_str(v)
    return (v == true) and "true" or "false"
end

local function default_state()
    return {
        player_hp = 0, player_hp_max = 1,
        enemy_hp  = 0, enemy_hp_max  = 1,
        player_ap = 0,
        enemy_break = 0, enemy_break_max = 1,
        in_battle = false, is_offensive_phase = false,
    }
end

-- ── State collection ──────────────────────────────────────────────────────────
local function collect_state()
    local s = default_state()

    local player = FindFirstOf(CONFIG.player_class)
    if player and player:IsValid() then
        s.player_hp     = safe_read(player, CONFIG.player_hp_prop)     or 0
        s.player_hp_max = safe_read(player, CONFIG.player_hp_max_prop) or 1
        s.player_ap     = safe_read(player, CONFIG.player_ap_prop)     or 0
    end

    local enemy = FindFirstOf(CONFIG.enemy_class)
    if enemy and enemy:IsValid() then
        s.enemy_hp        = safe_read(enemy, CONFIG.enemy_hp_prop)        or 0
        s.enemy_hp_max    = safe_read(enemy, CONFIG.enemy_hp_max_prop)    or 1
        s.enemy_break     = safe_read(enemy, CONFIG.enemy_break_prop)     or 0
        s.enemy_break_max = safe_read(enemy, CONFIG.enemy_break_max_prop) or 1
    end

    local battle = FindFirstOf(CONFIG.battle_class)
    if battle and battle:IsValid() then
        s.in_battle            = safe_read(battle, CONFIG.in_battle_prop)            or false
        s.is_offensive_phase   = safe_read(battle, CONFIG.is_offensive_phase_prop)   or false
    end

    return s
end

local function to_json(s)
    return string.format(
        '{"player_hp":%g,"player_hp_max":%g,'
        .. '"enemy_hp":%g,"enemy_hp_max":%g,'
        .. '"player_ap":%d,'
        .. '"enemy_break":%g,"enemy_break_max":%g,'
        .. '"in_battle":%s,"is_offensive_phase":%s}',
        s.player_hp, s.player_hp_max,
        s.enemy_hp,  s.enemy_hp_max,
        math.floor(s.player_ap or 0),
        s.enemy_break, s.enemy_break_max,
        bool_to_str(s.in_battle),
        bool_to_str(s.is_offensive_phase)
    )
end

-- ── Tick hook ─────────────────────────────────────────────────────────────────
local min_interval = CONFIG.write_hz > 0 and (1.0 / CONFIG.write_hz) or 0.0
local last_write   = 0.0

-- Hook the PlayerController tick as a reliable per-frame trigger.
-- Replace "BP_PlayerController_C" if the class name differs in this game.
RegisterHook("/Script/Engine.Actor:ReceiveTick", function(self, delta_seconds)
    -- Rate-limit writes
    local now = os.clock()
    if (now - last_write) < min_interval then return end
    last_write = now

    local state = collect_state()
    local json  = to_json(state)

    local f = io.open(OUTPUT_FILE, "w")
    if f then
        f:write(json)
        f:close()
    end
end)
