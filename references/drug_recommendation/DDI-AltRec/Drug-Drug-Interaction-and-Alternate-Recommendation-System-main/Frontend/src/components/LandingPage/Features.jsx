const Features = () => {
    const featureList = [
        { title: "AI-Powered Analysis", description: "Get insights using advanced AI algorithms for better decision-making." },
        { title: "Seamless Integration", description: "Easily integrate with popular tools and services for streamlined workflows." },
        { title: "Real-Time Monitoring", description: "Track data in real-time with intuitive dashboards and alerts." },
        { title: "Customizable Reports", description: "Generate tailored reports to suit your business needs and goals." },
    ];

    return (
        <div className="features-section"u id="features">
            <h2 className="features-heading">Features</h2>
            <div className="features-container">
                {featureList.map((feature, index) => (
                    <div key={index} className="feature-card">
                        <h3>{feature.title}</h3>
                        <p>{feature.description}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Features;