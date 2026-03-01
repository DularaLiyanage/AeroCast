import 'dart:convert';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart' as latlong;

// Update this to your backend IP
const String apiBase = "http://10.0.2.2:8000";

// Alternative tile providers (if OpenStreetMap blocks access):
// CartoDB: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
// Stadia Maps: 'https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png'

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
  runApp(const MaterialApp(
    home: SpatialScreen(),
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
  
  // Spatial specific data from second code
  String? spatialMethod;

  final MapController _mapCtrl = MapController();

  static const latlong.LatLng battaramullaLL =
      latlong.LatLng(6.901035, 79.926513);
  static const latlong.LatLng kandyLL =
      latlong.LatLng(7.292651, 80.635649);

  List<Marker> markers = [];

  @override
  void dispose() {
    dtCtrl.dispose();
    super.dispose();
  }

  Future<void> runForecast() async {
    final url = Uri.parse("$apiBase/spatial/predict"); // Using the spatial endpoint

    final body = {
      "station": station,
      "target": target,
      "datetime_str": dtCtrl.text.trim(),
      "overrides": overrides.isEmpty ? null : overrides,
    };

    try {
      print("Forecast request: $url");
      print("Forecast body: $body");
      final res = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(body),
      );
      print("Forecast response status: ${res.statusCode}");
      print("Forecast response body: ${res.body}");

      if (res.statusCode != 200) {
        throw Exception("HTTP ${res.statusCode}: ${res.body}");
      }

      final resp = jsonDecode(res.body);

      // Decoding spatial data
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
            child: const Icon(Icons.location_on, color: Colors.red, size: 40),
          ),
          Marker(
            point: kandyLL,
            child: const Icon(Icons.location_on, color: Colors.blue, size: 40),
          ),
        ];
      });
    } catch (e) {
      print("Forecast request failed: $e");
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Request failed: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = MediaQuery.of(context).size.width < 600;

    final initialOptions = MapOptions(
      initialCenter: const latlong.LatLng(7.05, 80.20),
      initialZoom: isMobile ? 7.0 : 8.0,
    );

    return Scaffold(
      appBar: AppBar(
        title: const Text("Aerocast - Air Quality Forecast"),
        backgroundColor: Colors.blue.shade700,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView( // Added to prevent overflow on small screens
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              // ------------------------- 
              // Controls Card
              // -------------------------
              Card(
                elevation: 3,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      DropdownButton<String>(
                        value: station,
                        isExpanded: true,
                        items: const [
                          DropdownMenuItem(value: "Battaramulla", child: Text("Battaramulla")),
                          DropdownMenuItem(value: "Kandy", child: Text("Kandy")),
                        ],
                        onChanged: (v) => setState(() => station = v!),
                      ),
                      const SizedBox(height: 12),
                      DropdownButton<String>(
                        value: target,
                        isExpanded: true,
                        items: const ["PM25", "PM10", "NO2", "SO2", "O3", "CO"]
                            .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                            .toList(),
                        onChanged: (v) => setState(() => target = v!),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: dtCtrl,
                        decoration: const InputDecoration(
                          labelText: "Date & Time (YYYY-MM-DD HH:MM:SS)",
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        onPressed: runForecast,
                        icon: const Icon(Icons.analytics),
                        label: const Text("Generate Prediction"),
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(double.infinity, 45),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // ------------------------- 
              // Results View (Integrated from code 2)
              // -------------------------
              if (prediction != null)
                Card(
                  color: Colors.green.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Predicted Value: ${prediction!.toStringAsFixed(2)}",
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                        ),
                        if (spatialMethod != null)
                          Text("Method: $spatialMethod"),
                        if (forecastTime != null)
                          Text("Forecast Time: $forecastTime", style: const TextStyle(color: Colors.grey)),
                      ],
                    ),
                  ),
                ),

              const SizedBox(height: 16),

              // ------------------------- 
              // Heatmap and Map Layout
              // -------------------------
              SizedBox(
                height: 400, // Fixed height for the map/heatmap area
                child: isMobile
                    ? Column(
                        children: [
                          Expanded(child: _buildHeatmapWidget()),
                          const SizedBox(height: 16),
                          Expanded(child: _buildMapWidget(initialOptions)),
                        ],
                      )
                    : Row(
                        children: [
                          Expanded(child: _buildHeatmapWidget()),
                          const SizedBox(width: 16),
                          Expanded(child: _buildMapWidget(initialOptions)),
                        ],
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeatmapWidget() {
    return GestureDetector(
      onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => HeatmapPage(heatmapBytes: heatmapBytes, target: target))),
      child: Card(
        child: Column(
          children: [
            const ListTile(leading: Icon(Icons.map), title: Text("Spatial Heatmap")),
            Expanded(
              child: heatmapBytes == null
                  ? const Center(child: Text("No Data Generated"))
                  : Image.memory(heatmapBytes!, fit: BoxFit.contain),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMapWidget(MapOptions options) {
    return GestureDetector(
      onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => MapPage(markers: markers, target: target, datetimeStr: dtCtrl.text))),
      child: Card(
        child: Column(
          children: [
            const ListTile(leading: Icon(Icons.location_on), title: Text("Interactive Map")),
            Expanded(
              child: FlutterMap(
                mapController: _mapCtrl,
                options: options,
                children: [
                  TileLayer(
                    urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                    userAgentPackageName: 'lk.edu.aerocast.project.v1',
                    tileProvider: UserAgentTileProvider(), // Custom provider with proper HTTP headers
                  ),
                  // Alternative: Use CartoDB if OSM blocks access
                  // urlTemplate: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                  // subdomains: const ['a', 'b', 'c', 'd'],
                  MarkerLayer(markers: markers),
                ],
              ),
            ),
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
      appBar: AppBar(title: Text("Heatmap - $target")),
      body: heatmapBytes == null 
          ? const Center(child: Text("No heatmap data")) 
          : InteractiveViewer(child: Center(child: Image.memory(heatmapBytes!))),
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
    if (isHoverLoading) return; // Prevent multiple simultaneous requests

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
      print("Hover request: $url");
      final res = await http.get(url);
      print("Hover response status: ${res.statusCode}");
      print("Hover response body: ${res.body}");

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
        print("Hover API error: ${res.statusCode} - ${res.body}");
        setState(() {
          hoverMethod = "error";
        });
      }
    } catch (e) {
      print("Hover request failed: $e");
      setState(() {
        hoverMethod = "error";
      });
    } finally {
      setState(() {
        isHoverLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Map - ${widget.target}")),
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
                tileProvider: UserAgentTileProvider(), // Custom provider with proper HTTP headers
              ),
              // Alternative: Use CartoDB if OSM blocks access
              // urlTemplate: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
              // subdomains: const ['a', 'b', 'c', 'd'],
              MarkerLayer(markers: widget.markers),
            ],
          ),
          if (hoverLatLng != null && hoverScreenPosition != null)
            Positioned(
              left: (hoverScreenPosition!.dx + 20 > MediaQuery.of(context).size.width - 200)
                  ? hoverScreenPosition!.dx - 220  // Position to the left if too close to right edge
                  : hoverScreenPosition!.dx + 20,   // Default: 20px to the right
              top: (hoverScreenPosition!.dy + 20 > MediaQuery.of(context).size.height - 150)
                  ? hoverScreenPosition!.dy - 170  // Position above if too close to bottom
                  : hoverScreenPosition!.dy + 20,   // Default: 20px below
              child: Card(
                color: Colors.white.withOpacity(0.95),
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        "${hoverLatLng!.latitude.toStringAsFixed(4)}, ${hoverLatLng!.longitude.toStringAsFixed(4)}",
                        style: const TextStyle(fontSize: 12, color: Colors.grey),
                      ),
                      const SizedBox(height: 4),
                      if (isHoverLoading)
                        const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            SizedBox(
                              width: 12, height: 12,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                            SizedBox(width: 8),
                            Text("Loading...", style: TextStyle(fontSize: 14)),
                          ],
                        )
                      else if (hoverMethod == "ensemble" && hoverValue != null)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              "${widget.target}: ${hoverValue!.toStringAsFixed(2)}",
                              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                            ),
                            Text(
                              "Method: $hoverMethod",
                              style: const TextStyle(fontSize: 12, color: Colors.grey),
                            ),
                          ],
                        )
                      else if (hoverMethod == "out_of_range")
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Text(
                              "Out of Range",
                              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.red),
                            ),
                            Text(
                              "Distance: ${hoverDistanceKm?.toStringAsFixed(1)} km",
                              style: const TextStyle(fontSize: 12, color: Colors.grey),
                            ),
                          ],
                        )
                      else
                        const Text(
                          "Error loading value",
                          style: TextStyle(fontSize: 14, color: Colors.red),
                        ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}