import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart' as latlong;

// Update this to your backend IP
const String apiBase = "http://10.0.2.2:8000";

// OpenStreetMap User-Agent: Custom TileProvider with proper HTTP headers for OSM compliance
class UserAgentTileProvider extends TileProvider {
  final http.Client _client = http.Client();

  @override
  ImageProvider getImage(TileCoordinates coordinates, TileLayer options) {
    final url = getTileUrl(coordinates, options);
    return NetworkImage(
      url,
      headers: {
        'User-Agent': 'AerocastAirQualityApp/1.0 (lk.edu.aerocast.project.v1)',
        'Referer': 'https://aerocast.project.edu',
        'Accept': 'image/png,image/*,*/*',
      },
    );
  }

  @override
  void dispose() {
    _client.close();
    super.dispose();
  }
}

void main() {
  runApp(MaterialApp(
    theme: ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueAccent),
      appBarTheme: const AppBarTheme(
        centerTitle: true,
        elevation: 0,
      ),
    ),
    home: const SpatialScreen(),
    debugShowCheckedModeBanner: false,
  ));
}

class SpatialScreen extends StatefulWidget {
  const SpatialScreen({super.key});

  @override
  State<SpatialScreen> createState() => _SpatialScreenState();
}

class _SpatialScreenState extends State<SpatialScreen> {
  // Form values
  String station = "Battaramulla";
  String target = "PM25";
  late final TextEditingController dtCtrl;

  Map<String, dynamic> overrides = {};

  // Response fields
  double? prediction;
  String? forecastTime;
  Uint8List? heatmapBytes;
  
  // Spatial specific data
  String? spatialMethod;

  bool isLoading = false;

  static const latlong.LatLng battaramullaLL = latlong.LatLng(6.901035, 79.926513);
  static const latlong.LatLng kandyLL = latlong.LatLng(7.292651, 80.635649);

  List<Marker> markers = [];

  @override
  void initState() {
    super.initState();
    final now = DateTime.now();
    final formattedDate =
        "${now.year.toString().padLeft(4, '0')}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')} ${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}:00";
    dtCtrl = TextEditingController(text: formattedDate);
  }

  @override
  void dispose() {
    dtCtrl.dispose();
    super.dispose();
  }

  Future<void> runForecast() async {
    setState(() {
      isLoading = true;
      prediction = null;
    });

    final url = Uri.parse("$apiBase/spatial/predict");

    final body = {
      "station": station,
      "target": target,
      "datetime_str": dtCtrl.text.trim(),
      "overrides": overrides.isEmpty ? null : overrides,
    };

    try {
      final res = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(body),
      );

      if (res.statusCode != 200) {
        throw Exception("HTTP ${res.statusCode}: ${res.body}");
      }

      final resp = jsonDecode(res.body);
      final spatialInterpolation = resp["spatial_interpolation"];
      final b64 = resp["heatmap_png_base64"] as String?;
      final bytes = b64 != null ? base64Decode(b64) : null;

      setState(() {
        prediction = (resp["prediction"] as num?)?.toDouble();
        
        // Add 1 hour to the user's selected time to accurately reflect the 1-Hr Ahead prediction
        try {
          DateTime parsed = DateTime.parse(dtCtrl.text);
          DateTime ahead = parsed.add(const Duration(hours: 1));
          forecastTime = "${ahead.year.toString().padLeft(4, '0')}-${ahead.month.toString().padLeft(2, '0')}-${ahead.day.toString().padLeft(2, '0')} ${ahead.hour.toString().padLeft(2, '0')}:${ahead.minute.toString().padLeft(2, '0')}";
        } catch(e) {
          forecastTime = resp["forecast_time"];
        }
        
        heatmapBytes = bytes;

        if (spatialInterpolation != null) {
          spatialMethod = spatialInterpolation["method"];
        }

        markers = [
          Marker(
            point: battaramullaLL,
            child: const Icon(Icons.location_on, color: Colors.redAccent, size: 40),
          ),
          Marker(
            point: kandyLL,
            child: const Icon(Icons.location_on, color: Colors.blueAccent, size: 40),
          ),
        ];
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text("Request failed: $e"),
          backgroundColor: Colors.redAccent,
          behavior: SnackBarBehavior.floating,
        ),
      );
    } finally {
      if (mounted) {
        setState(() {
          isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade50,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 200.0,
            floating: false,
            pinned: true,
            title: const Text(
              "Spatial Forecast",
              style: TextStyle(fontWeight: FontWeight.w600, color: Colors.white, fontSize: 24),
            ),
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.blue.shade800, Colors.blue.shade400],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: Stack(
                  children: [
                    Positioned(
                      right: -20,
                      bottom: -20,
                      child: Icon(Icons.map_outlined, size: 150, color: Colors.white.withValues(alpha: 0.15)),
                    ),
                    Positioned(
                      left: 24,
                      bottom: 30,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            "AIR QUALITY MAPPING",
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: Colors.white.withValues(alpha: 0.85),
                              letterSpacing: 1.2,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            "Predict precise pollutant levels across\nnearby regions.",
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w400,
                              color: Colors.white.withValues(alpha: 0.75),
                              height: 1.4,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildInputsCard(),
                  const SizedBox(height: 24),
                  if (isLoading)
                    const Center(child: CircularProgressIndicator())
                  else if (prediction != null) ...[
                    _buildPredictionCard(),
                    const SizedBox(height: 24),
                    const Text(
                      "Visualizations",
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87),
                    ),
                    const SizedBox(height: 12),
                    _buildActionButtons(context),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _selectDateTime(BuildContext context) async {
    final DateTime? pickedDate = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime.now().subtract(const Duration(days: 30)),
      lastDate: DateTime.now().add(const Duration(days: 30)),
    );
    if (pickedDate != null) {
      if (!context.mounted) return;
      final TimeOfDay? pickedTime = await showTimePicker(
        context: context,
        initialTime: TimeOfDay.now(),
      );
      if (pickedTime != null) {
        setState(() {
          dtCtrl.text =
              "${pickedDate.year.toString().padLeft(4, '0')}-${pickedDate.month.toString().padLeft(2, '0')}-${pickedDate.day.toString().padLeft(2, '0')} ${pickedTime.hour.toString().padLeft(2, '0')}:${pickedTime.minute.toString().padLeft(2, '0')}:00";
        });
      }
    }
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 24, color: Colors.blueGrey.shade800),
        const SizedBox(width: 12),
        Text(title, style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: Colors.blueGrey.shade900)),
        const SizedBox(width: 16),
        Expanded(child: Divider(color: Colors.grey.shade300, thickness: 1)),
      ],
    );
  }

  Widget _buildInputsCard() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(32),
        border: Border.all(color: Colors.blue.shade50, width: 2),
        boxShadow: [
          BoxShadow(
            color: Colors.blueGrey.withValues(alpha: 0.06),
            blurRadius: 24,
            offset: const Offset(0, 12),
          )
        ],
      ),
      padding: const EdgeInsets.all(28),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionHeader("Station", Icons.location_on_rounded),
          const SizedBox(height: 20),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: ["Battaramulla", "Kandy"].map((s) {
                final isSelected = station == s;
                return GestureDetector(
                  onTap: () => setState(() => station = s),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 250),
                    margin: const EdgeInsets.only(right: 12),
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                    decoration: BoxDecoration(
                      color: isSelected ? Colors.blue.shade600 : Colors.transparent,
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(color: isSelected ? Colors.blue.shade600 : Colors.grey.shade300, width: 1.5),
                      boxShadow: isSelected 
                        ? [BoxShadow(color: Colors.blue.shade300.withValues(alpha: 0.5), blurRadius: 10, offset: const Offset(0, 4))] 
                        : [],
                    ),
                    child: Center(
                      child: Text(
                        s, 
                        style: TextStyle(
                          color: isSelected ? Colors.white : Colors.blueGrey.shade600, 
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.w600,
                          fontSize: 15,
                        ),
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
          ),
          
          const SizedBox(height: 40),
          _buildSectionHeader("Pollutant Target", Icons.blur_on_rounded),
          const SizedBox(height: 20),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: ["PM25", "PM10", "NO2", "SO2", "O3", "CO"].map((p) {
              final isSelected = target == p;
              return GestureDetector(
                onTap: () => setState(() => target = p),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  decoration: BoxDecoration(
                    color: isSelected ? Colors.blue.shade600 : Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: isSelected ? Colors.blue.shade600 : Colors.grey.shade200,
                      width: 1.5,
                    ),
                    boxShadow: isSelected
                        ? [BoxShadow(color: Colors.blue.withValues(alpha: 0.3), blurRadius: 8, offset: const Offset(0, 4))]
                        : [BoxShadow(color: Colors.black.withValues(alpha: 0.02), blurRadius: 4, offset: const Offset(0, 2))],
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (isSelected) 
                        const Padding(
                          padding: EdgeInsets.only(right: 6),
                          child: Icon(Icons.check_circle_rounded, color: Colors.white, size: 16),
                        ),
                      Text(
                        p,
                        style: TextStyle(
                          color: isSelected ? Colors.white : Colors.blueGrey.shade700,
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.w600,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }).toList(),
          ),

          const SizedBox(height: 40),
          _buildSectionHeader("Date & Time", Icons.access_time_filled_rounded),
          const SizedBox(height: 8),
          InkWell(
            onTap: () => _selectDateTime(context),
            borderRadius: BorderRadius.circular(8),
            splashColor: Colors.blue.withValues(alpha: 0.1),
            highlightColor: Colors.transparent,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 12.0, horizontal: 8.0),
              child: Row(
                children: [
                  Icon(Icons.calendar_today_rounded, size: 20, color: Colors.blue.shade600),
                  const SizedBox(width: 14),
                  Builder(
                    builder: (context) {
                      final parts = dtCtrl.text.split(' ');
                      final dateStr = parts.isNotEmpty ? parts[0] : '';
                      String timeStr = parts.length > 1 ? parts[1] : '';
                      if (timeStr.length >= 5) {
                        timeStr = timeStr.substring(0, 5);
                      }
                      
                      return RichText(
                        text: TextSpan(
                          children: [
                            TextSpan(
                              text: dateStr,
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w500,
                                color: Color(0xFF334155),
                                letterSpacing: 0.2,
                              ),
                            ),
                            const TextSpan(text: "   "),
                            TextSpan(
                              text: timeStr,
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w400,
                                color: Color(0xFF64748B),
                                letterSpacing: 0.1,
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
          
          const SizedBox(height: 48),
          GestureDetector(
            onTap: isLoading ? null : runForecast,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 250),
              width: double.infinity,
              height: 64,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: isLoading 
                    ? [Colors.grey.shade400, Colors.grey.shade400] 
                    : [Colors.blue.shade600, Colors.blue.shade400],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(20),
                boxShadow: isLoading 
                  ? [] 
                  : [BoxShadow(color: Colors.blue.shade400.withValues(alpha: 0.4), blurRadius: 16, offset: const Offset(0, 8))],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (isLoading)
                    const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5)),
                  if (isLoading) const SizedBox(width: 12),
                  Text(
                    isLoading ? "Analyzing Data..." : "Generate Prediction", 
                    style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white, letterSpacing: 0.5)
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPredictionCard() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.teal.shade500, Colors.teal.shade800],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(
            color: Colors.teal.withValues(alpha: 0.4),
            blurRadius: 16,
            offset: const Offset(0, 8),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.2),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.auto_graph_rounded, color: Colors.white, size: 22),
              ),
              const SizedBox(width: 12),
              Text(
                "Prediction Ready",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white.withValues(alpha: 0.95)),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(30),
                  border: Border.all(color: Colors.white.withValues(alpha: 0.3), width: 1),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.update_rounded, color: Colors.white70, size: 14),
                    const SizedBox(width: 6),
                    Text(
                      "1-Hr Ahead",
                      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.white.withValues(alpha: 0.85), letterSpacing: 0.5),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          Text(
            prediction!.toStringAsFixed(2),
            style: const TextStyle(fontSize: 56, fontWeight: FontWeight.w900, color: Colors.white, letterSpacing: -1.5),
          ),
          Text(
            "Predicted Value (${target})",
            style: TextStyle(fontSize: 15, fontWeight: FontWeight.w500, color: Colors.white.withValues(alpha: 0.75)),
          ),
          const SizedBox(height: 20),
          if (forecastTime != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.access_time_rounded, color: Colors.white70, size: 18),
                  const SizedBox(width: 8),
                  Text("Valid For:", style: TextStyle(fontSize: 13, color: Colors.white.withValues(alpha: 0.8))),
                  const SizedBox(width: 8),
                  Text(forecastTime!, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white)),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    final isMobile = MediaQuery.of(context).size.width < 600;
    
    Widget heatmapButton = _ActionCard(
      title: "View Spatial Heatmap",
      subtitle: "High-resolution 2D pollutant distribution",
      icon: Icons.grid_on_rounded,
      color: Colors.deepPurpleAccent,
      onTap: () {
        if (heatmapBytes != null) {
          Navigator.push(context, MaterialPageRoute(builder: (_) => HeatmapPage(heatmapBytes: heatmapBytes, target: target)));
        } else {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("No heatmap data available")));
        }
      },
    );

    Widget interactiveMapButton = _ActionCard(
      title: "Explore Interactive Map",
      subtitle: "Pinch, zoom, and tap locations for exact values",
      icon: Icons.map_rounded,
      color: Colors.orangeAccent,
      onTap: () {
        Navigator.push(context, MaterialPageRoute(builder: (_) => MapPage(markers: markers, target: target, datetimeStr: dtCtrl.text)));
      },
    );

    if (isMobile) {
      return Column(
        children: [
          heatmapButton,
          const SizedBox(height: 12),
          interactiveMapButton,
        ],
      );
    } else {
      return Row(
        children: [
          Expanded(child: heatmapButton),
          const SizedBox(width: 16),
          Expanded(child: interactiveMapButton),
        ],
      );
    }
  }
}

class _ActionCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _ActionCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.grey.shade200),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.1),
              blurRadius: 10,
              offset: const Offset(0, 4),
            )
          ],
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: color, size: 28),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.black87)),
                  const SizedBox(height: 4),
                  Text(subtitle, style: TextStyle(fontSize: 13, color: Colors.grey.shade600)),
                ],
              ),
            ),
            Icon(Icons.chevron_right, color: Colors.grey.shade400),
          ],
        ),
      ),
    );
  }
}

// =========================
// Full-Screen Heatmap Page
// =========================
class HeatmapPage extends StatelessWidget {
  final Uint8List? heatmapBytes;
  final String target;
  
  const HeatmapPage({super.key, this.heatmapBytes, required this.target});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade50, // Matches the app's ash background
      appBar: AppBar(
        title: Text(
          "Heatmap - $target", 
          style: const TextStyle(color: Colors.black87, fontWeight: FontWeight.bold)
        ),
        backgroundColor: Colors.white.withValues(alpha: 0.9),
        iconTheme: const IconThemeData(color: Colors.black87),
        elevation: 0,
        centerTitle: true,
      ),
      body: heatmapBytes == null 
          ? Center(child: Text("No heatmap data generated.", style: TextStyle(color: Colors.grey.shade600))) 
          : Center(
              child: Container(
                margin: const EdgeInsets.all(16.0),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.05),
                      blurRadius: 20,
                      offset: const Offset(0, 10),
                    )
                  ],
                ),
                clipBehavior: Clip.antiAlias,
                child: InteractiveViewer(
                  panEnabled: true,
                  minScale: 1.0,
                  maxScale: 4.0,
                  child: Hero(
                    tag: 'heatmap_image',
                    child: Image.memory(
                      heatmapBytes!,
                      fit: BoxFit.contain,
                    ),
                  ),
                ),
              ),
            ),
    );
  }
}

// =========================
// Full-Screen Map Page (with Hover functionality)
// =========================
class MapPage extends StatefulWidget {
  final List<Marker> markers;
  final String target;
  final String datetimeStr;
  
  const MapPage({super.key, required this.markers, required this.target, required this.datetimeStr});

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  double? hoverValue;
  latlong.LatLng? hoverLatLng;
  String? hoverMethod;
  double? hoverDistanceKm;
  bool isHoverLoading = false;
  
  final MapController _mapCtrl = MapController();
  Offset? hoverScreenPosition;

  @override
  void dispose() {
    _mapCtrl.dispose();
    super.dispose();
  }

  Future<void> _getHoverValue(TapPosition tapPosition, latlong.LatLng point) async {
    if (isHoverLoading) return;

    setState(() {
      isHoverLoading = true;
      hoverLatLng = point;
      hoverScreenPosition = tapPosition.relative;
      hoverValue = null;
      hoverMethod = null;
      hoverDistanceKm = null;
    });

    final url = Uri.parse("$apiBase/spatial/hover_value?lat=${point.latitude}&lon=${point.longitude}&target=${widget.target}&datetime_str=${Uri.encodeComponent(widget.datetimeStr)}");
    try {
      final res = await http.get(url);
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        setState(() {
          hoverMethod = data["method"];
          if (hoverMethod == "ensemble") {
            hoverValue = (data["value"] as num?)?.toDouble();
            hoverDistanceKm = null;
          } else if (hoverMethod == "out_of_range") {
            hoverValue = null;
            hoverDistanceKm = (data["distance_km"] as num?)?.toDouble();
          }
        });
      } else {
        setState(() { hoverMethod = "error"; });
      }
    } catch (e) {
      setState(() { hoverMethod = "error"; });
    } finally {
      if (mounted) setState(() { isHoverLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.9),
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 10)
            ],
          ),
          child: Text(
            "Interactive Map - ${widget.target}",
            style: const TextStyle(color: Colors.black87, fontSize: 16, fontWeight: FontWeight.bold),
          ),
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.black87),
        centerTitle: true,
      ),
      body: Stack(
        children: [
          FlutterMap(
            mapController: _mapCtrl,
            options: MapOptions(
              initialCenter: const latlong.LatLng(7.05, 80.20),
              initialZoom: 8.0,
              onTap: (tapPos, latlng) => _getHoverValue(tapPos, latlng),
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'lk.edu.aerocast.project.v1',
                tileProvider: UserAgentTileProvider(),
              ),
              MarkerLayer(markers: widget.markers),
            ],
          ),
          if (hoverLatLng != null && hoverScreenPosition != null)
            _buildHoverCard(context),
        ],
      ),
    );
  }

  Widget _buildHoverCard(BuildContext context) {
    // Dynamic positioning
    double leftPos = hoverScreenPosition!.dx + 20;
    double topPos = hoverScreenPosition!.dy + 20;
    final screenWidth = MediaQuery.of(context).size.width;
    final screenHeight = MediaQuery.of(context).size.height;

    if (leftPos + 220 > screenWidth) leftPos = hoverScreenPosition!.dx - 240;
    if (topPos + 150 > screenHeight) topPos = hoverScreenPosition!.dy - 160;

    return Positioned(
      left: leftPos,
      top: topPos,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Container(
            width: 220,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.85),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white.withOpacity(0.4)),
              boxShadow: [
                BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 20, spreadRadius: -5)
              ]
            ),
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  children: [
                    const Icon(Icons.location_on, size: 16, color: Colors.blueAccent),
                    const SizedBox(width: 4),
                    Text(
                      "${hoverLatLng!.latitude.toStringAsFixed(3)}, ${hoverLatLng!.longitude.toStringAsFixed(3)}",
                      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.grey.shade700),
                    ),
                  ],
                ),
                const Divider(),
                if (isHoverLoading)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 8.0),
                    child: Row(
                      children: [
                        SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
                        SizedBox(width: 12),
                        Text("Calculating...", style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500)),
                      ],
                    ),
                  )
                else if (hoverMethod == "ensemble" && hoverValue != null)
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.baseline,
                        textBaseline: TextBaseline.alphabetic,
                        children: [
                          Text(
                            hoverValue!.toStringAsFixed(1),
                            style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.black87),
                          ),
                          const SizedBox(width: 4),
                          Text(widget.target, style: const TextStyle(fontSize: 14, color: Colors.black54)),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(color: Colors.blue.withOpacity(0.1), borderRadius: BorderRadius.circular(4)),
                        child: Text("Method: $hoverMethod", style: TextStyle(fontSize: 10, color: Colors.blue.shade700)),
                      ),
                    ],
                  )
                else if (hoverMethod == "out_of_range")
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.warning_amber_rounded, color: Colors.deepOrange, size: 18),
                          SizedBox(width: 6),
                          Text("Out of Range", style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.deepOrange)),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text("Dist: ${hoverDistanceKm?.toStringAsFixed(1)} km", style: const TextStyle(fontSize: 12, color: Colors.black54)),
                    ],
                  )
                else
                  const Text("Error fetching data", style: TextStyle(fontSize: 14, color: Colors.redAccent)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}