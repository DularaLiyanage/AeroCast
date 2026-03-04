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
  final TextEditingController dtCtrl =
      TextEditingController(text: "2024-12-31 12:00:00");

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
        forecastTime = resp["forecast_time"];
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
            expandedHeight: 160.0,
            floating: false,
            pinned: true,
            flexibleSpace: FlexibleSpaceBar(
              title: const Text(
                "Spatial Forecast",
                style: TextStyle(fontWeight: FontWeight.w600, color: Colors.white),
              ),
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
                      top: -20,
                      child: Icon(Icons.map_outlined, size: 140, color: Colors.white.withOpacity(0.15)),
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

  Widget _buildInputsCard() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 10,
            offset: const Offset(0, 4),
          )
        ],
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Configuration",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87),
          ),
          const SizedBox(height: 16),
          _buildDropdownField(
            label: "Station",
            icon: Icons.location_city,
            value: station,
            items: const ["Battaramulla", "Kandy"],
            onChanged: (v) => setState(() => station = v!),
          ),
          const SizedBox(height: 16),
          _buildDropdownField(
            label: "Pollutant Target",
            icon: Icons.cloud_outlined,
            value: target,
            items: const ["PM25", "PM10", "NO2", "SO2", "O3", "CO"],
            onChanged: (v) => setState(() => target = v!),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: dtCtrl,
            decoration: InputDecoration(
              labelText: "Date & Time",
              hintText: "YYYY-MM-DD HH:MM:SS",
              prefixIcon: const Icon(Icons.calendar_today_outlined, color: Colors.blueAccent),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.grey.shade300),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.grey.shade300),
              ),
              filled: true,
              fillColor: Colors.grey.shade50,
            ),
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            height: 50,
            child: FilledButton.icon(
              onPressed: isLoading ? null : runForecast,
              icon: const Icon(Icons.analytics_outlined),
              label: const Text("Generate Prediction", style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              style: FilledButton.styleFrom(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                backgroundColor: Colors.blue.shade700,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDropdownField({
    required String label,
    required IconData icon,
    required String value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, color: Colors.blueAccent),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        filled: true,
        fillColor: Colors.grey.shade50,
      ),
      items: items.map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
      onChanged: onChanged,
    );
  }

  Widget _buildPredictionCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.teal.shade400, Colors.teal.shade700],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.teal.withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 6),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.check_circle_outline, color: Colors.white, size: 28),
              const SizedBox(width: 8),
              Text(
                "Prediction Ready",
                style: TextStyle(fontSize: 16, color: Colors.white.withOpacity(0.9)),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            prediction!.toStringAsFixed(2),
            style: const TextStyle(fontSize: 48, fontWeight: FontWeight.bold, color: Colors.white),
          ),
          const Text(
            "Predicted Value",
            style: TextStyle(fontSize: 14, color: Colors.white70),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (spatialMethod != null)
                  Text("Method: $spatialMethod", style: const TextStyle(color: Colors.white)),
                if (forecastTime != null)
                  Text("Time: $forecastTime", style: const TextStyle(color: Colors.white)),
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
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text("Heatmap - $target", style: const TextStyle(color: Colors.white)),
        backgroundColor: Colors.transparent,
        iconTheme: const IconThemeData(color: Colors.white),
        elevation: 0,
      ),
      body: heatmapBytes == null 
          ? const Center(child: Text("No heatmap data generated.", style: TextStyle(color: Colors.white54))) 
          : Center(
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