import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'detailed_report_screen.dart';
import '../providers/aqi_provider.dart';
import '../utils/constants.dart';
import '../widgets/aqi_chart.dart';
import '../widgets/custom_card.dart';

class DashboardScreen extends StatefulWidget {
  final String? initialLocation;

  const DashboardScreen({super.key, this.initialLocation});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  late String _currentLocation;
  bool _showAqi = true;
  bool _showMin = true;
  bool _showMax = true;

  @override
  void initState() {
    super.initState();
    _currentLocation = widget.initialLocation ?? 'Battaramulla';
    // Fetch data after the first frame
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<AqiProvider>(context, listen: false)
          .loadPredictions(_currentLocation);
    });
  }

  void _switchLocation(String location) {
    final provider = Provider.of<AqiProvider>(context, listen: false);
    if (provider.isLoading || _currentLocation == location) return;

    setState(() {
      _currentLocation = location;
    });
    provider.loadPredictions(location);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text('Air Quality Risk',
            style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.primaryText,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(60),
          child: Consumer<AqiProvider>(
            builder: (context, provider, _) => Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              child: Row(
                children: [
                  Expanded(
                    child: _buildLocationButton('Battaramulla',
                        isLoading: provider.isLoading),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildLocationButton('Kandy',
                        isLoading: provider.isLoading),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
      body: Consumer<AqiProvider>(
        builder: (context, provider, child) {
          if (provider.isLoading) {
            return _buildLoadingSkeleton();
          }

          if (provider.error != null) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline,
                      size: 48, color: Colors.redAccent),
                  const SizedBox(height: 16),
                  Text('Error loading data',
                      style: GoogleFonts.poppins(fontSize: 18)),
                  Text(provider.error!,
                      style: GoogleFonts.poppins(color: Colors.grey),
                      textAlign: TextAlign.center),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: () => provider.loadPredictions(_currentLocation),
                    child: const Text('Try Again'),
                  )
                ],
              ),
            );
          }

          if (provider.predictions.isEmpty) {
            return const Center(child: Text("No data available."));
          }

          // Data is ready
          final current = provider.predictions.first;
          final predictions = provider.predictions;

          return AnimatedSwitcher(
            duration: const Duration(milliseconds: 280),
            switchInCurve: Curves.easeOutCubic,
            switchOutCurve: Curves.easeInCubic,
            child: SingleChildScrollView(
              key: ValueKey(_currentLocation),
              physics: const BouncingScrollPhysics(),
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 0. Health Alert Banner
                    if (current.healthAlert != null)
                      Builder(builder: (context) {
                        final alertColorName =
                            current.healthAlert!['color'] as String?;
                        final alertColor = alertColorName != null
                            ? AppColors.getColor(alertColorName)
                            : AppColors.getColor(current.colour);

                        return Container(
                          width: double.infinity,
                          margin: const EdgeInsets.only(bottom: 24),
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: alertColor.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: alertColor, width: 1.5),
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.warning_amber_rounded,
                                  color: alertColor, size: 32),
                              const SizedBox(width: 16),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      current.healthAlert!['title'] ?? 'Alert',
                                      style: GoogleFonts.poppins(
                                          fontWeight: FontWeight.bold,
                                          color: alertColor,
                                          fontSize: 16),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      current.healthAlert!['message'] ?? '',
                                      style: GoogleFonts.poppins(
                                          color: AppColors.primaryText,
                                          fontSize: 13),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ).animate().fadeIn().slideY(begin: -0.5);
                      }),

                    // 1. Hero Status Card
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: AppColors.getColor(current.colour),
                        borderRadius: BorderRadius.circular(32),
                        boxShadow: [
                          BoxShadow(
                            color: AppColors.getColor(current.colour)
                                .withOpacity(0.4),
                            blurRadius: 20,
                            offset: const Offset(0, 10),
                          )
                        ],
                      ),
                      child: Column(
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                'Current Status',
                                style: GoogleFonts.poppins(
                                    color: Colors.white70, fontSize: 14),
                              ),
                              Text(
                                "${current.date} • ${current.time}",
                                style: GoogleFonts.poppins(
                                    color: Colors.white60, fontSize: 12),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            current.aqi.toInt().toString(),
                            style: GoogleFonts.poppins(
                              color: Colors.white,
                              fontSize: 72,
                              fontWeight: FontWeight.bold,
                              height: 1,
                            ),
                          ),
                          Text(
                            'AQI',
                            style: GoogleFonts.poppins(
                                color: Colors.white54, fontSize: 16),
                          ),
                          const SizedBox(height: 16),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 16, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              current.riskLevel.toUpperCase(),
                              style: GoogleFonts.poppins(
                                color: Colors.white,
                                fontWeight: FontWeight.w600,
                                letterSpacing: 1.2,
                              ),
                            ),
                          ),
                        ],
                      ),
                    )
                        .animate()
                        .scale(curve: Curves.easeOutBack, duration: 600.ms),

                    const SizedBox(height: 24),

                    // 1.1 Quick Stats Strip
                    Row(
                      children: [
                        Expanded(
                          child: _buildStatPill(
                            icon: Icons.show_chart_rounded,
                            label: 'Min',
                            value: current.min.toStringAsFixed(1),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _buildStatPill(
                            icon: Icons.trending_up_rounded,
                            label: 'Max',
                            value: current.max.toStringAsFixed(1),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _buildStatPill(
                            icon: Icons.schedule_rounded,
                            label: 'Updated',
                            value: current.time,
                          ),
                        ),
                      ],
                    ).animate().fadeIn(delay: 120.ms),

                    const SizedBox(height: 24),

                    // 2. Reasoning Card (Moved Up)
                    CustomCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(Icons.lightbulb_outline,
                                  color: AppColors.getColor(current.colour)),
                              const SizedBox(width: 8),
                              Text(
                                'Insight',
                                style: GoogleFonts.poppins(
                                    fontSize: 16, fontWeight: FontWeight.bold),
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),
                          Text(
                            current.reasoning,
                            style: GoogleFonts.poppins(
                              color: AppColors.secondaryText,
                              fontSize: 15,
                              height: 1.5,
                            ),
                          ),
                        ],
                      ),
                    ).animate().fadeIn(delay: 200.ms).slideY(),

                    const SizedBox(height: 32),

                    // 3. Chart Section
                    Text(
                      '24-Hour Forecast',
                      style: GoogleFonts.poppins(
                          fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        _buildChartToggleChip(
                          label: 'AQI',
                          isActive: _showAqi,
                          onTap: () => setState(() => _showAqi = !_showAqi),
                        ),
                        _buildChartToggleChip(
                          label: 'Min',
                          isActive: _showMin,
                          onTap: () => setState(() => _showMin = !_showMin),
                        ),
                        _buildChartToggleChip(
                          label: 'Max',
                          isActive: _showMax,
                          onTap: () => setState(() => _showMax = !_showMax),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Container(
                      height: 250,
                      padding: const EdgeInsets.all(0),
                      child: AqiChart(
                        predictions: predictions,
                        showAqi: _showAqi,
                        showMin: _showMin,
                        showMax: _showMax,
                      ),
                    ).animate().fadeIn(duration: 800.ms).slideX(),

                    const SizedBox(height: 24),

                    Center(
                      child: OutlinedButton.icon(
                        onPressed: () {
                          Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (_) => DetailedReportScreen(
                                      predictions: predictions,
                                      location: _currentLocation)));
                        },
                        icon: const Icon(Icons.list_alt),
                        label: const Text("View Detailed Report"),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: AppColors.primaryText,
                          side: BorderSide(
                              color: AppColors.primaryText.withOpacity(0.2)),
                          padding: const EdgeInsets.symmetric(
                              horizontal: 24, vertical: 12),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20)),
                        ),
                      ),
                    ),

                    const SizedBox(height: 32),

                    const SizedBox(height: 32),

                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildChartToggleChip({
    required String label,
    required bool isActive,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: isActive ? AppColors.primaryText : Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isActive
                ? AppColors.primaryText
                : AppColors.primaryText.withOpacity(0.15),
          ),
          boxShadow: isActive ? [AppStyles.softShadow] : null,
        ),
        child: Text(
          label,
          style: GoogleFonts.poppins(
            color: isActive ? Colors.white : AppColors.primaryText,
            fontWeight: FontWeight.w600,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  Widget _buildLocationButton(String location, {required bool isLoading}) {
    final isSelected = _currentLocation == location;
    return GestureDetector(
      onTap: () => _switchLocation(location),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color:
              isSelected ? AppColors.primaryText : Colors.grey.withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected
                ? AppColors.primaryText
                : Colors.grey.withOpacity(0.3),
            width: 1,
          ),
        ),
        child: Center(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                location,
                style: GoogleFonts.poppins(
                  color: isSelected ? Colors.white : AppColors.primaryText,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                  fontSize: 14,
                ),
              ),
              if (isSelected && isLoading) ...[
                const SizedBox(width: 8),
                SizedBox(
                  width: 12,
                  height: 12,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor:
                        const AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatPill({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [AppStyles.softShadow],
      ),
      child: Column(
        children: [
          Icon(icon, size: 16, color: AppColors.secondaryText),
          const SizedBox(height: 6),
          Text(
            value,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.poppins(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: AppColors.primaryText,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: GoogleFonts.poppins(
              fontSize: 10,
              color: AppColors.secondaryText,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingSkeleton() {
    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _skeletonBox(height: 160, radius: 28),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(child: _skeletonBox(height: 72, radius: 16)),
                const SizedBox(width: 10),
                Expanded(child: _skeletonBox(height: 72, radius: 16)),
                const SizedBox(width: 10),
                Expanded(child: _skeletonBox(height: 72, radius: 16)),
              ],
            ),
            const SizedBox(height: 20),
            _skeletonBox(height: 130, radius: 24),
            const SizedBox(height: 24),
            _skeletonBox(height: 24, widthFactor: 0.45, radius: 12),
            const SizedBox(height: 16),
            _skeletonBox(height: 240, radius: 20),
            const SizedBox(height: 24),
            _skeletonBox(height: 46, widthFactor: 0.6, radius: 18),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _skeletonBox({
    required double height,
    double radius = 12,
    double widthFactor = 1,
  }) {
    final base = Colors.grey.shade300;
    return FractionallySizedBox(
      widthFactor: widthFactor,
      child: Container(
        height: height,
        decoration: BoxDecoration(
          color: base,
          borderRadius: BorderRadius.circular(radius),
        ),
      )
          .animate(onPlay: (controller) => controller.repeat(reverse: true))
          .fade(begin: 0.5, end: 1.0, duration: 700.ms),
    );
  }
}
